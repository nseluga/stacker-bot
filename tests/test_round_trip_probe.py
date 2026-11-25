"""
Gates for scripts/round_trip_probe.py (stage plan specialist_explore, actor-throughput stage
piece 1b, steps 13c–13e): in both message designs, and with requests split over two servers,
every answer reaches the actor, hand and decision that asked for it through the real process
queues; with a net in each actor, every answer matches its decision scored alone, with each
weight-copy design too (step 13f); the learner's write reaches every shared net; design C's
actors refresh on schedule and redo a copy a write overlapped; and the run's gain is capped by
the learner as slowed by the server.
"""

import sys
from pathlib import Path

import multiprocessing

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import round_trip_probe
import throughput_ceiling
from test_net import DEV_SEED

PLAYERS = 4
POOL_SIZE = 32
NUM_ACTORS = 2
HANDS = 4
# The engine cost of the ceiling split, rounded; any small cost works here.
KEPT_SECONDS = 129e-6
# Every decision a choice turn: the most replies to check per second run.
CHOICE_SHARE = 1.0
WARMUP_SECONDS = 0.5
MEASURE_SECONDS = 2.0
# Two servers: the fewest that split the actors, one each.
TWO_SERVERS = 2

# run_speedup known answers: a learner cap of 3.56, halved by a learner running at half speed.
LEARNER_CAP = 3.56
LEARNER_SLOWDOWN = 2.0
HIGH_ACTOR_SPEEDUP = 3.0
LOW_ACTOR_SPEEDUP = 1.2

# Weight-copy tests: two CPU nets stand in for teapot's two GPUs.
TWO_DEVICES = ["cpu", "cpu"]
# Any shift works: it only has to make the learner's weights differ from the starting ones.
WEIGHT_SHIFT = 1.0
REFRESH_EVERY = 10
# Two and a half refresh periods: two refreshes, then updates the actor must not take yet.
REFRESH_TEST_UPDATES = 25
# One whole write finishing during the actor's copy raises the version by 2 (see write_weights).
VERSION_STEPS_PER_WRITE = 2


def pool_and_answers_alone():
    """
    The dev decision pool, and each decision's answer when the net scores it alone: the
    reference every routed or batched answer must match.
    """
    decisions = throughput_ceiling.choice_decisions(PLAYERS, DEV_SEED, POOL_SIZE)
    net = throughput_ceiling.untrained_net()
    alone = []
    for decision in decisions:
        alone.append(throughput_ceiling.batched_choices(net, [decision], torch.device("cpu"))[0])
    # With one answer for every decision, a misrouted answer would still look right.
    assert len(set(alone)) > 1
    return decisions, alone


@pytest.mark.parametrize("bulk, num_servers", [(False, 1), (True, 1), (True, TWO_SERVERS)])
def test_each_actor_gets_its_own_answers(bulk, num_servers):
    # catches: a reply sent to the wrong actor (both actors' request ids start at 0, so only
    # the answer's value shows it), to the wrong slot (the actor raises and never reports), and
    # answers out of row order within a batch, in each message design; the bulk flag not
    # reaching the actors (one request per message either way); and with two servers, every
    # actor sent to one server (servers_used 1), or a server replying through the wrong queue.
    decisions, alone = pool_and_answers_alone()

    result = round_trip_probe.run_setting(
        NUM_ACTORS, HANDS, False, decisions, KEPT_SECONDS, CHOICE_SHARE, "cpu", DEV_SEED,
        WARMUP_SECONDS, MEASURE_SECONDS, True, bulk, num_servers,
    )
    assert result["servers_used"] == num_servers
    # With one decision per call, row order could not go wrong.
    assert result["mean_batch"] > 1
    if bulk:
        assert result["requests_per_message"] > 1
    else:
        assert result["requests_per_message"] == 1
    for actor_answers in result["answers"]:
        assert len(actor_answers) > 0
        for pool_index, answer in actor_answers:
            assert answer == alone[pool_index]


@pytest.mark.parametrize(
    "weight_copy",
    [
        round_trip_probe.WEIGHT_COPY_NONE,
        round_trip_probe.WEIGHT_COPY_SHARED,
        round_trip_probe.WEIGHT_COPY_HYBRID,
    ],
)
def test_actor_net_answers_match(weight_copy):
    # catches: an actor-net answer filed under the wrong pool index or row of its batch (the
    # answers stop matching); hands not batched into one call (mean batch 1); a server started
    # in the actor-net design (servers_used not 0); with designs B and C, an actor scoring with
    # a net that is not the starting weights the shared nets hold (learner off, so nothing
    # writes them), or a shared net that does not reach the actor process.
    decisions, alone = pool_and_answers_alone()
    result = round_trip_probe.run_setting(
        NUM_ACTORS, HANDS, False, decisions, KEPT_SECONDS, CHOICE_SHARE, "cpu", DEV_SEED,
        WARMUP_SECONDS, MEASURE_SECONDS, True, False, 0, 1, True, weight_copy,
    )
    assert result["servers_used"] == 0
    assert result["mean_batch"] > 1
    for actor_answers in result["answers"]:
        assert len(actor_answers) > 0
        for pool_index, answer in actor_answers:
            assert answer == alone[pool_index]


def shifted_learner():
    """An untrained net with every weight shifted by WEIGHT_SHIFT: a learner after an update."""
    learner = throughput_ceiling.untrained_net()
    with torch.no_grad():
        for parameter in learner.parameters():
            parameter.add_(WEIGHT_SHIFT)
    return learner


def nets_equal(first_net, second_net):
    """True if every weight of the two nets is exactly equal."""
    for first, second in zip(first_net.parameters(), second_net.parameters()):
        if not torch.equal(first, second):
            return False
    return True


def test_write_reaches_every_shared_net():
    # catches: the learner writing only one shared net (the second GPU's actors would score
    # with stale weights forever), a write into new tensors instead of the shared ones (the
    # actors' handles would miss it), and the version not left even and raised per write.
    shared_nets, version = round_trip_probe.make_shared_nets(
        multiprocessing.get_context("spawn"), TWO_DEVICES
    )
    handles = []
    for shared_net in shared_nets:
        handles.append(next(shared_net.parameters()))
    learner = shifted_learner()

    round_trip_probe.write_weights(learner, shared_nets, version)

    assert version.value == VERSION_STEPS_PER_WRITE
    for net_index in range(len(shared_nets)):
        assert nets_equal(shared_nets[net_index], learner)
        assert next(shared_nets[net_index].parameters()) is handles[net_index]


def test_refresh_follows_its_schedule():
    # catches: refresh_every ignored (a copy at every update), a refresh that never copies, and
    # a lag that grows past refresh_every.
    shared_nets, version = round_trip_probe.make_shared_nets(
        multiprocessing.get_context("spawn"), ["cpu"]
    )
    learner = throughput_ceiling.untrained_net()
    private_net = throughput_ceiling.untrained_net()
    held_updates = 0
    for update in range(1, REFRESH_TEST_UPDATES + 1):
        with torch.no_grad():
            for parameter in learner.parameters():
                parameter.add_(WEIGHT_SHIFT)
        round_trip_probe.write_weights(learner, shared_nets, version)
        held_updates, copy_seconds = round_trip_probe.refresh_private_net(
            private_net, shared_nets[0], version, held_updates, REFRESH_EVERY
        )
        # The actor takes the weights exactly at each multiple of REFRESH_EVERY, and only then.
        if update % REFRESH_EVERY == 0:
            assert held_updates == update
            assert nets_equal(private_net, learner)
        else:
            assert held_updates == update - update % REFRESH_EVERY
            assert not nets_equal(private_net, learner)
            assert copy_seconds == 0.0
        assert update - held_updates < REFRESH_EVERY


def test_refresh_redoes_a_copy_a_write_overlapped():
    # catches: a copy kept although a write landed during it (it may mix old and new weights),
    # and the held updates taken from the version before the copy instead of after.
    shared_nets, version = round_trip_probe.make_shared_nets(
        multiprocessing.get_context("spawn"), ["cpu"]
    )
    shared_net = shared_nets[0]
    learner = shifted_learner()
    round_trip_probe.write_weights(learner, shared_nets, version)
    private_net = throughput_ceiling.untrained_net()

    # The first time the actor reads the shared weights, one whole learner write lands at once.
    reads = []
    read_shared_weights = shared_net.state_dict

    def state_dict_with_a_write():
        reads.append(True)
        weights = read_shared_weights()
        if len(reads) == 1:
            round_trip_probe.write_weights(shifted_learner(), shared_nets, version)
        return weights

    shared_net.state_dict = state_dict_with_a_write
    held_updates, copy_seconds = round_trip_probe.refresh_private_net(
        private_net, shared_net, version, 0, 1
    )

    assert len(reads) == 2
    assert held_updates == version.value // VERSION_STEPS_PER_WRITE


def test_run_speedup_known_answer():
    # catches: the cap not divided by the learner's slowdown, or multiplied by it (either gives
    # 3.0), and min swapped for max (gives 3.0 and 1.78).
    assert round_trip_probe.run_speedup(
        HIGH_ACTOR_SPEEDUP, LEARNER_CAP, LEARNER_SLOWDOWN
    ) == pytest.approx(LEARNER_CAP / LEARNER_SLOWDOWN)
    assert round_trip_probe.run_speedup(
        LOW_ACTOR_SPEEDUP, LEARNER_CAP, LEARNER_SLOWDOWN
    ) == pytest.approx(LOW_ACTOR_SPEEDUP)
