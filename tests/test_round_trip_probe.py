"""
Gates for scripts/round_trip_probe.py (stage plan specialist_explore, actor-throughput stage
piece 1b, steps 13c–13e): in both message designs, and with requests split over two servers,
every answer reaches the actor, hand and decision that asked for it through the real process
queues; with a net in each actor, every answer matches its decision scored alone; and the
run's gain is capped by the learner as slowed by the server.
"""

import sys
from pathlib import Path

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


def test_actor_net_answers_match():
    # catches: an actor-net answer filed under the wrong pool index or row of its batch (the
    # answers stop matching); hands not batched into one call (mean batch 1); a server started
    # in the actor-net design (servers_used not 0).
    decisions, alone = pool_and_answers_alone()
    result = round_trip_probe.run_setting(
        NUM_ACTORS, HANDS, False, decisions, KEPT_SECONDS, CHOICE_SHARE, "cpu", DEV_SEED,
        WARMUP_SECONDS, MEASURE_SECONDS, True, False, 0, 1, True,
    )
    assert result["servers_used"] == 0
    assert result["mean_batch"] > 1
    for actor_answers in result["answers"]:
        assert len(actor_answers) > 0
        for pool_index, answer in actor_answers:
            assert answer == alone[pool_index]


def test_run_speedup_known_answer():
    # catches: the cap not divided by the learner's slowdown, or multiplied by it (either gives
    # 3.0), and min swapped for max (gives 3.0 and 1.78).
    assert round_trip_probe.run_speedup(
        HIGH_ACTOR_SPEEDUP, LEARNER_CAP, LEARNER_SLOWDOWN
    ) == pytest.approx(LEARNER_CAP / LEARNER_SLOWDOWN)
    assert round_trip_probe.run_speedup(
        LOW_ACTOR_SPEEDUP, LEARNER_CAP, LEARNER_SLOWDOWN
    ) == pytest.approx(LOW_ACTOR_SPEEDUP)
