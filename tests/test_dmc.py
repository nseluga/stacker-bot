"""
Gates for the M3 Deep Monte-Carlo loop (stage plan specialist_explore, stage 3 `DMC loop`,
pieces 2-6 one process, pieces 8-10 many processes).

The pick: a forced turn never calls the net; epsilon 0 takes the highest Q; epsilon 1 is uniform.
The samples: every decision of a hand is stored, forced ones included; each stored action row is
the action that was played; each return is the deciding seat's own final score.
The batch: one action row per sample (K = 1), returns with one axis.
The update: the loss falls on a fixed batch; one step moves every parameter; a gradient longer
than MAX_GRAD_NORM is cut to exactly that length; gradients left from an earlier update are
cleared first.
The optimizer: build_optimizer gives the optimizer and the rate it was asked for, with the
spec's other settings (part 17), and refuses a name it does not know.
The arm arguments: train hands its sizes to both nets, its optimizer name and rate to the
learner, and its epsilon to every actor.
The loop: the same seed gives the same result; another seed does not; no sample is trained on
twice.
The streams: a training hand's rng name uses the seed, the actor and the hand number, and is
never the name of an evaluation stream. An actor's hands are the ones its stream names say.
Many processes: two actors both deliver samples, the actors' net ends equal to the learner's,
the last checkpoint holds the learner's weights, and a used run directory is refused.
The split: the training code never names the sealed deals; the script refuses seeds 0-2.
The device: on a machine with no CUDA GPU the script refuses --device cuda before it makes a run.
On a machine with one (teapot): an update on the GPU gives the CPU's loss; after a GPU run the
learner is on the GPU, the actors' net is on the CPU in shared memory with the same weights, and
every checkpoint is held on the CPU.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import argparse
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import parse_cards
from test_net import DEV_SEED, real_decisions, seeded_net

# scripts/ is not a package, so the folder goes on the import path for train_dmc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_dmc  # noqa: E402
from evaluate_checkpoints import load_checkpoint_net  # noqa: E402

from stacker.dmc import (
    EPSILON,
    FAST_ACTOR_HANDS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_PLAYERS,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    HandSlot,
    actor_loop,
    advance_slots,
    build_optimizer,
    choose_action,
    choose_actions_batched,
    collate_samples,
    draw_chair_seat,
    fast_actor_loop,
    hand_stream_name,
    learner_step,
    make_shared_nets,
    self_play_hand,
    shuffle_returns,
    train,
    train_one_process,
)
from stacker.encode import ACTION_DIM, HISTORY_WINDOW, MOVE_DIM, STATE_DIM, encode_action
from stacker.game import HandState, new_hand, play_out
from stacker.heuristic import heuristic_agent
from stacker.net import HEAD_CLASSIFIER, QNet

# Batched-pick gate: copies of one dev hand's decisions, each copy with its own rngs, so the
# pool is large enough that both explore and greedy picks happen at epsilon 0.3. Pairs of
# decisions share an rng, as the decisions of one hand do across passes.
BATCHED_POOL_COPIES = 4
BATCHED_EPSILON = 0.3
DECISIONS_PER_RNG = 2
# Every real Q pushed far below 0, so a padded slot (Q 0) wins the argmax if the mask is lost.
NEGATIVE_BIAS = -100.0

# Lockstep gate: 8 hands of one actor's stream, at the batched gate's epsilon so both explore
# and greedy picks happen.
LOCKSTEP_HANDS = 8
LOCKSTEP_SEED = 10
LOCKSTEP_ACTOR = 0

# Fast-actor gate: two real actor processes on the CPU, a few slots each, read until 24 hands.
FAST_GATE_ACTORS = 2
FAST_GATE_HANDS = 4
FAST_GATE_ITEMS = 24
FAST_GATE_TIMEOUT_SECONDS = 60

# Fast-run gate: enough updates that an actor stuck on update 0 lags by far more than a
# working one, whose lag is bounded by the hands waiting in the queue.
FAST_RUN_UPDATES = 20

# A rate that is not the control's 1e-4, so a rate that build_optimizer drops shows up.
ARM_LEARNING_RATE = 3e-4
# An arm net for the pass-through gate: width 256, 3 hidden layers and summary size 256, all
# off the control. Its weight count from spec part 16: 379,904 (LSTM) + 606,720 + 2 x 65,792
# + 257.
ARM_WIDTH = 256
ARM_HIDDEN_LAYERS = 3
ARM_HISTORY_HIDDEN = 256
ARM_PARAMETER_COUNT = 1_118_465
# The same arm net with the classifier head: its last layer has 3 outputs instead of 1, so 2 more
# rows of ARM_WIDTH weights and 2 more biases.
ARM_CLASSIFIER_PARAMETER_COUNT = ARM_PARAMETER_COUNT + 2 * (ARM_WIDTH + 1)
# An untrained classifier's loss on real samples is within this of ln 3 (spec part 18).
CLASSIFIER_INITIAL_LOSS_TOLERANCE = 0.01
# The whole-hand window arm (decision log 2026-11-22).
ARM_HISTORY_WINDOW = 96
# Arm tag of the script gate's tiny run; it ends the run directory's name.
SCRIPT_ARM_TAG = "script-flags-gate"
# Run name of that gate's base: a tiny control run made just before the arm.
SCRIPT_BASE_RUN = "script-flags-gate-base"
# Legal actions of the decision the pick tests use.
PICK_NUM_ACTIONS = 5
# Hand-set Q for those actions: the highest is at index 3, not at either end.
PICK_Q_VALUES = (0.1, -0.4, 0.3, 0.9, 0.2)
PICK_BEST_INDEX = 3
# Uniform-pick test: 3,000 draws over 5 actions expect 600 each, standard deviation about 22.
# The band is about 4.5 standard deviations wide each way, and the rng is seeded.
UNIFORM_DRAWS = 3000
UNIFORM_COUNT_LOW = 500
UNIFORM_COUNT_HIGH = 700
# Scripted hand: one card per seat, seat 0 leads the highest card, so every turn is forced.
# Seat 0 plays and is out (2 points); seats 1, 2, 3 cannot beat it and pass; seat 1 leads the next
# round and is out (1 point). The deciding seats, in order, and the return each must get:
SCRIPTED_HANDS = ("2s", "5d", "4d", "3d")
SCRIPTED_SEATS = (0, 1, 2, 3, 1)
SCRIPTED_RETURNS = (2.0, 1.0, 0.0, 0.0, 1.0)
# Loss-falls test: repeated updates on one hand's samples. One-off run: 1.01 -> 0.007 in 30.
FIXED_BATCH_UPDATES = 30
FIXED_BATCH_LOSS_FRACTION = 0.1
# Clip test: returns are multiplied by this so the gradient is far longer than MAX_GRAD_NORM.
LARGE_RETURN_SCALE = 1000.0
CLIP_TOLERANCE = 1e-3
# Stale-gradient test: the value written into every stored gradient before the update. Over the
# net's 2.3 million weights that is a gradient about 1,500 long, far above a real one.
STALE_GRADIENT_VALUE = 1.0
STALE_GRADIENT_TOLERANCE = 1e-6
# Determinism test: a small loop, about a second per run.
LOOP_UPDATES = 3
LOOP_SAMPLES_PER_UPDATE = 64
# Stream-name tests: any actor id and hand number; each is changed by one to get a neighbour.
STREAM_ACTOR = 3
STREAM_HAND = 7
# Actor-stream gate: actors compared, and the hands each plays before the stand-in queue stops it.
STREAM_GATE_ACTORS = 2
STREAM_GATE_HANDS = 2
# Heuristic-chair gates: the seat the hand tests give the heuristic (any seat but 0 would do; 0
# leads more often, so a seat that is not first is the stricter case).
CHAIR_TEST_SEAT = 2
# Seat-draw test: 400 hand streams over 4 seats expect 100 each, standard deviation about 8.7.
# The band is about 4.6 standard deviations wide each way, and the streams are fixed.
CHAIR_DRAWS = 400
CHAIR_COUNT_LOW = 60
CHAIR_COUNT_HIGH = 140
# Two-actor smoke: a hand is about 70 decisions, so 256 samples take several hands and three
# such batches leave time for both actors to deliver. About 3 s, most of it process start.
SMOKE_ACTORS = 2
SMOKE_UPDATES = 3
SMOKE_SAMPLES_PER_UPDATE = 256
# train works the variance out in 32-bit floats; the test in Python's 64-bit ones.
VARIANCE_TOLERANCE = 1e-6
# The rng stream name train gives the label-shuffle control for this seed (spec part 13).
SHUFFLE_STREAM = f"shuffle:{DEV_SEED}"
# The largest variance a batch of returns can have: values lie in [0, 2], and the spread is
# widest with half at each end, (1 - 0)² = 1.
MAX_RETURN_VARIANCE = 1.0
# The files whose text must never name the sealed deals (rule 1), relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CODE_FILES = ("src/stacker/dmc.py", "scripts/train_dmc.py")
# A test seed (rule 2: 0-2 are for confirm runs) and the run name the refused run would take.
REFUSED_SEED = 2
REFUSED_RUN_NAME = "gate_refused_test_seed"
# The largest gap allowed between one update's loss on the GPU and on the CPU. The two do the
# same 32-bit sums in a different order, so the last digits differ; a wrong batch or a wrong
# net gives a gap thousands of times larger.
DEVICE_LOSS_TOLERANCE = 1e-5
# Marks a gate that needs a GPU. Checked when pytest collects the file.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU here")
# The run name a run refused for its device would take.
NO_GPU_RUN_NAME = "gate_refused_no_gpu"


def raising_net(*net_inputs):
    """Stands in for a QNet on turns where the net must not be called: fails the test if it is."""
    raise AssertionError("the net was called")


def fixed_q_net(q_row):
    """
    A stand-in for a QNet that ignores its inputs.
    q_row: sequence of K floats. Returns a function giving them as a float32 [1, K] tensor, the
    shape QNet.forward gives for a batch of one decision.
    """

    def net(*net_inputs):
        return torch.tensor([q_row])

    return net


def decision_with_actions(num_actions):
    """
    A real encoded decision cut down to its first num_actions legal actions.
    Returns (state, history, mask, action_matrix [num_actions, ACTION_DIM]).
    """
    for state, history, mask, action_matrix in real_decisions(NUM_PLAYERS):
        if len(action_matrix) >= num_actions:
            return state, history, mask, action_matrix[:num_actions]
    raise AssertionError(f"no decision with {num_actions} legal actions in the dev hand")


def dealt_hand_samples():
    """
    Plays one dev-seed hand with a fresh seeded net in every seat.
    Returns (samples, trace) as self_play_hand gives them.
    """
    rng = random.Random(DEV_SEED)
    state = new_hand(NUM_PLAYERS, rng)
    return self_play_hand(state, seeded_net(), EPSILON, rng)


class QueueFull(Exception):
    """Raised by StoppingQueue to end an actor_loop call, which never returns by itself."""


class StoppingQueue:
    """
    Stands in for the actors' sample queue in a test that runs actor_loop in this process.
    Keeps every item put on it in self.items, and raises QueueFull on the put that brings the
    count to max_items, after storing that item.
    """

    def __init__(self, max_items):
        self.max_items = max_items
        self.items = []

    def put(self, item):
        self.items.append(item)
        if len(self.items) == self.max_items:
            raise QueueFull


def samples_equal(first, second):
    """
    True when two lists of samples from self_play_hand hold the same decisions in the same
    order: equal state vectors, histories, history masks, taken actions and returns.
    """
    if len(first) != len(second):
        return False
    for first_sample, second_sample in zip(first, second):
        # The first four parts are numpy arrays; the last is the return, a float.
        for part_index in range(4):
            if not np.array_equal(first_sample[part_index], second_sample[part_index]):
                return False
        if first_sample[4] != second_sample[4]:
            return False
    return True


def returns_of(samples):
    """The return of each sample in a list of samples, in order: a list of floats."""
    returns = []
    for sample in samples:
        returns.append(sample[4])
    return returns


def recorded_train(monkeypatch, run_dir, shuffle_control):
    """
    Runs the smoke-sized train once with recorders in place of learner_step and around
    shuffle_returns, so a test sees what train handed each of them.
    run_dir: a path that does not exist yet. shuffle_control: train's flag.
    Returns (shuffle_calls, learner_batches, log_lines):
      shuffle_calls: one (samples handed in, samples returned) pair per call of shuffle_returns.
      learner_batches: the batch learner_step was handed at each update.
      log_lines: the run's log.jsonl, one dict per update.
    The recording learner_step updates nothing, so the net stays untrained; the batches and
    the log are what is under test.
    """
    shuffle_calls = []
    learner_batches = []

    def recording_shuffle(samples, rng):
        shuffled = shuffle_returns(samples, rng)
        shuffle_calls.append((samples, shuffled))
        return shuffled

    def recording_learner_step(q_net, optimizer, samples):
        learner_batches.append(samples)
        return 0.0, 0.0

    monkeypatch.setattr("stacker.dmc.shuffle_returns", recording_shuffle)
    monkeypatch.setattr("stacker.dmc.learner_step", recording_learner_step)
    run_dir.mkdir()
    train(
        DEV_SEED,
        SMOKE_ACTORS,
        run_dir,
        SMOKE_UPDATES,
        samples_per_update=SMOKE_SAMPLES_PER_UPDATE,
        checkpoint_every=SMOKE_UPDATES,
        shuffle_control=shuffle_control,
    )

    log_lines = []
    with open(run_dir / "log.jsonl") as log_file:
        for line in log_file:
            log_lines.append(json.loads(line))
    return shuffle_calls, learner_batches, log_lines


def control_optimizer(q_net):
    """The control RMSprop over q_net's parameters, with the settings train_one_process uses."""
    return torch.optim.RMSprop(
        q_net.parameters(),
        lr=LEARNING_RATE,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        momentum=RMSPROP_MOMENTUM,
    )


def gradient_norm(q_net):
    """The length of q_net's stored gradients, all parameters taken as one long vector."""
    squared_sum = 0.0
    for parameter in q_net.parameters():
        squared_sum += float((parameter.grad**2).sum())
    return squared_sum**0.5


def test_forced_turn_skips_the_net():
    decision = decision_with_actions(1)
    rng = random.Random(DEV_SEED)
    state_before = rng.getstate()
    # Epsilon 0 never explores, so without the forced-turn rule the net would be called.
    index = choose_action(raising_net, decision, 0.0, rng)
    assert index == 0
    # A forced turn draws no random number: the rng is exactly where it was.
    assert rng.getstate() == state_before


def test_epsilon_zero_picks_the_highest_q():
    decision = decision_with_actions(PICK_NUM_ACTIONS)
    net = fixed_q_net(PICK_Q_VALUES)
    index = choose_action(net, decision, 0.0, random.Random(DEV_SEED))
    assert index == PICK_BEST_INDEX


def test_epsilon_one_picks_every_action_about_equally():
    decision = decision_with_actions(PICK_NUM_ACTIONS)
    rng = random.Random(DEV_SEED)
    counts = [0] * PICK_NUM_ACTIONS
    for _ in range(UNIFORM_DRAWS):
        # Epsilon 1 always explores, so the net must never be called.
        index = choose_action(raising_net, decision, 1.0, rng)
        counts[index] += 1
    for count in counts:
        assert UNIFORM_COUNT_LOW <= count <= UNIFORM_COUNT_HIGH


def test_batched_picks_equal_choose_action():
    # catches: a padded slot winning the argmax (mask lost), an extra or missing rng draw (a
    # draw on a forced turn, the explore draw skipped), and answers put back at the wrong place.
    one_hand = real_decisions(NUM_PLAYERS)
    decisions = []
    for _ in range(BATCHED_POOL_COPIES):
        decisions.extend(one_hand)
    action_counts = set()
    for decision in decisions:
        action_counts.add(len(decision[3]))
    # Forced turns and more than one K with a choice: otherwise a branch goes untested.
    assert 1 in action_counts
    assert len(action_counts) > 2

    net = seeded_net()
    with torch.no_grad():
        net.mlp[-1].bias.fill_(NEGATIVE_BIAS)

    # Two identical sets of rngs: one for the batched call, one for choose_action.
    batched_rngs = []
    alone_rngs = []
    for position in range(len(decisions)):
        name = f"batched-pick:{position // DECISIONS_PER_RNG}"
        if position % DECISIONS_PER_RNG == 0:
            batched_rngs.append(random.Random(name))
            alone_rngs.append(random.Random(name))
        else:
            batched_rngs.append(batched_rngs[-1])
            alone_rngs.append(alone_rngs[-1])

    batched = choose_actions_batched(
        net, decisions, BATCHED_EPSILON, batched_rngs, torch.device("cpu")
    )
    alone = []
    for position in range(len(decisions)):
        alone.append(
            choose_action(net, decisions[position], BATCHED_EPSILON, alone_rngs[position])
        )

    assert batched == alone
    # Each rng ends where choose_action leaves it: the hand's next draws are unchanged.
    for position in range(len(decisions)):
        assert batched_rngs[position].getstate() == alone_rngs[position].getstate()


@needs_cuda
def test_lockstep_hands_on_gpu_equal_cpu_hands():
    # catches: the GPU's float math flipping a near-tie argmax, so a fast actor's hand on the
    # GPU is no longer the hand self_play_hand plays on the CPU from the same stream.
    cpu_net = seeded_net()
    gpu_net = seeded_net().to("cuda")
    slots = []
    for hand_index in range(LOCKSTEP_HANDS):
        rng = random.Random(hand_stream_name(LOCKSTEP_SEED, LOCKSTEP_ACTOR, hand_index))
        slots.append(HandSlot(hand_index, rng, new_hand(NUM_PLAYERS, rng)))

    gpu_traces = {}
    while len(gpu_traces) < LOCKSTEP_HANDS:
        finished = advance_slots(
            slots, gpu_net, BATCHED_EPSILON, HISTORY_WINDOW, torch.device("cuda")
        )
        for slot, _ in finished:
            gpu_traces[slot.hand_index] = slot.trace

    for hand_index in range(LOCKSTEP_HANDS):
        rng = random.Random(hand_stream_name(LOCKSTEP_SEED, LOCKSTEP_ACTOR, hand_index))
        _, cpu_trace = self_play_hand(new_hand(NUM_PLAYERS, rng), cpu_net, BATCHED_EPSILON, rng)
        assert gpu_traces[hand_index] == cpu_trace


def test_lockstep_hands_equal_self_play_hand():
    # catches: one hand's records landing in another (cross-talk), a decision given the wrong
    # seat's score, decisions out of order, and a hand that never finishes or finishes twice.
    net = seeded_net()
    device = torch.device("cpu")
    slots = []
    for hand_index in range(LOCKSTEP_HANDS):
        rng = random.Random(hand_stream_name(LOCKSTEP_SEED, LOCKSTEP_ACTOR, hand_index))
        slots.append(HandSlot(hand_index, rng, new_hand(NUM_PLAYERS, rng)))

    # Passes until every hand is over; each finished hand is kept by its hand index.
    finished_by_hand = {}
    while len(finished_by_hand) < LOCKSTEP_HANDS:
        finished = advance_slots(slots, net, BATCHED_EPSILON, HISTORY_WINDOW, device)
        # A pass with every hand already over would loop forever: some hand must be in play.
        assert len(finished) > 0 or any_in_play(slots)
        for slot, samples in finished:
            assert slot.hand_index not in finished_by_hand
            finished_by_hand[slot.hand_index] = (slot.trace, samples)

    for hand_index in range(LOCKSTEP_HANDS):
        rng = random.Random(hand_stream_name(LOCKSTEP_SEED, LOCKSTEP_ACTOR, hand_index))
        expected_samples, expected_trace = self_play_hand(
            new_hand(NUM_PLAYERS, rng), net, BATCHED_EPSILON, rng
        )
        trace, samples = finished_by_hand[hand_index]
        assert trace == expected_trace
        assert len(samples) == len(expected_samples)
        for sample, expected in zip(samples, expected_samples):
            # The four arrays exactly equal, then the return.
            for part in range(4):
                assert np.array_equal(sample[part], expected[part])
            assert sample[4] == expected[4]


def any_in_play(slots):
    """True if some slot's hand still has a seat to act."""
    for slot in slots:
        if slot.state.turn is not None:
            return True
    return False


def test_fast_actor_plays_its_own_streams():
    # catches: a hand index reused when a slot refills, two actors playing the same deal (the
    # actor id missing from the stream), an actor scoring with weights other than the shared
    # net's, and the queue item packed in the wrong order.
    net = seeded_net()
    context = torch.multiprocessing.get_context("spawn")
    device_names = []
    for _ in range(FAST_GATE_ACTORS):
        device_names.append("cpu")
    shared_nets, version = make_shared_nets(context, device_names, net)
    sample_queue = context.Queue()

    actors = []
    for actor_id in range(FAST_GATE_ACTORS):
        actor = context.Process(
            target=fast_actor_loop,
            args=(
                actor_id,
                LOCKSTEP_SEED,
                "cpu",
                shared_nets[actor_id],
                version,
                sample_queue,
                BATCHED_EPSILON,
                HISTORY_WINDOW,
                FAST_GATE_HANDS,
            ),
            daemon=True,
        )
        actor.start()
        actors.append(actor)
    items = []
    try:
        for _ in range(FAST_GATE_ITEMS):
            items.append(sample_queue.get(timeout=FAST_GATE_TIMEOUT_SECONDS))
    finally:
        for actor in actors:
            actor.terminate()
        for actor in actors:
            actor.join()

    seen = set()
    actors_seen = set()
    for actor_id, hand_index, held_updates, samples in items:
        assert (actor_id, hand_index) not in seen
        seen.add((actor_id, hand_index))
        actors_seen.add(actor_id)
        # No learner ran, so the version stayed 0 and the first refresh took update 0.
        assert held_updates == 0
        rng = random.Random(hand_stream_name(LOCKSTEP_SEED, actor_id, hand_index))
        expected, _ = self_play_hand(new_hand(NUM_PLAYERS, rng), net, BATCHED_EPSILON, rng)
        assert samples_equal(samples, expected)
    assert actors_seen == set(range(FAST_GATE_ACTORS))


def test_fast_actor_run_learns_from_fresh_weights(tmp_path):
    # catches: the actors' refresh skipped or write_weights never called (held_updates stays 0,
    # so the lag on the last update is the whole run), an actor given a net index past the
    # shared nets, and a checkpoint saved from weights older than the learner's.
    q_net, actor_net = train(
        DEV_SEED,
        SMOKE_ACTORS,
        tmp_path,
        FAST_RUN_UPDATES,
        samples_per_update=SMOKE_SAMPLES_PER_UPDATE,
        checkpoint_every=FAST_RUN_UPDATES,
        fast_actor=True,
        actor_hands=FAST_GATE_HANDS,
        actor_devices=("cpu", "cpu"),
    )

    log_lines = []
    with open(tmp_path / "log.jsonl") as log_file:
        for line in log_file:
            log_lines.append(json.loads(line))
    assert len(log_lines) == FAST_RUN_UPDATES
    for log_line in log_lines:
        assert math.isfinite(log_line["loss"])
        # A hand cannot hold weights newer than the learner's.
        assert log_line["actor_lag_max"] >= 0

    # Before the last update the learner had made FAST_RUN_UPDATES - 1 updates; an actor that
    # never took new weights would lag by exactly that on every hand.
    assert log_lines[-1]["actor_lag_mean"] < FAST_RUN_UPDATES - 1

    # The last checkpoint holds the learner's final weights.
    checkpoint = torch.load(tmp_path / f"checkpoint_{FAST_RUN_UPDATES:06d}.pt")
    learner_weights = q_net.state_dict()
    for name in learner_weights:
        assert torch.equal(checkpoint[name], learner_weights[name].cpu())


def test_scripted_hand_gives_each_decision_its_own_seats_score():
    hands = []
    for hand_text in SCRIPTED_HANDS:
        hands.append(parse_cards(hand_text))
    state = HandState(hands, 1, 0, None, [])

    # Every turn is forced, so the net is never needed and epsilon does not matter.
    samples, trace = self_play_hand(state, raising_net, EPSILON, random.Random(DEV_SEED))

    seats = []
    for seat, _ in trace:
        seats.append(seat)
    assert tuple(seats) == SCRIPTED_SEATS

    returns = []
    for sample in samples:
        returns.append(sample[4])
    # Seat 1 decides twice (a pass, then its last card): both decisions get its 1 point.
    assert tuple(returns) == SCRIPTED_RETURNS


def test_samples_match_the_hand_that_was_played():
    samples, trace = dealt_hand_samples()

    # One sample per decision, forced turns included.
    assert len(samples) == len(trace)

    # The return each seat's decisions carry, filled in as the samples are walked.
    returns_by_seat = {}
    for (seat, action), sample in zip(trace, samples):
        state_vector, history, history_mask, taken_action, hand_return = sample

        # Decode: the one stored action row is the encoding of the action in the trace.
        assert taken_action.shape == (1, ACTION_DIM)
        assert np.array_equal(taken_action[0], encode_action(action))

        # Every decision of one seat carries the same return.
        if seat in returns_by_seat:
            assert hand_return == returns_by_seat[seat]
        returns_by_seat[seat] = hand_return

    # Across the four seats the returns are one 2, one 1 and two 0s.
    assert sorted(returns_by_seat.values()) == [0.0, 0.0, 1.0, 2.0]
    # The hand ends on the play that puts the second seat out, so the last decider scored 1.
    last_seat = trace[-1][0]
    assert returns_by_seat[last_seat] == 1.0


def test_collate_samples_shapes():
    samples, _ = dealt_hand_samples()
    batch_size = len(samples)
    net_inputs, taken_index, returns = collate_samples(samples)
    state, history, history_mask, actions, action_mask = net_inputs

    assert state.shape == (batch_size, STATE_DIM)
    assert history.shape == (batch_size, HISTORY_WINDOW, MOVE_DIM)
    assert history_mask.shape == (batch_size, HISTORY_WINDOW)
    # K = 1: one action row per sample, none of it padding.
    assert actions.shape == (batch_size, 1, ACTION_DIM)
    assert action_mask.shape == (batch_size, 1)
    assert bool(action_mask.all())

    assert taken_index.shape == (batch_size,)
    assert taken_index.dtype == torch.int64
    assert bool((taken_index == 0).all())

    # One axis: a [B, 1] here would broadcast to [B, B] inside the loss.
    assert returns.shape == (batch_size,)
    assert returns.dtype == torch.float32
    for row, sample in enumerate(samples):
        assert np.array_equal(actions[row, 0].numpy(), sample[3][0])
        assert float(returns[row]) == sample[4]


def test_loss_falls_on_a_fixed_batch():
    samples, _ = dealt_hand_samples()
    net = seeded_net()
    optimizer = control_optimizer(net)
    losses = []
    for _ in range(FIXED_BATCH_UPDATES):
        loss, _ = learner_step(net, optimizer, samples)
        losses.append(loss)
    assert losses[-1] < FIXED_BATCH_LOSS_FRACTION * losses[0]


def test_untrained_classifier_loss_is_ln3():
    # catches: the learner training a classifier net's Q with the squared error (about 0.6 here)
    # instead of the cross-entropy on its logits, or a sum where the mean belongs.
    samples, _ = dealt_hand_samples()
    torch.manual_seed(DEV_SEED)
    net = QNet(head=HEAD_CLASSIFIER)
    loss, _ = learner_step(net, control_optimizer(net), samples)
    # Untrained logits are near 0, so every class gets about 1/3 and the loss is about ln 3,
    # whatever the labels.
    assert abs(loss - math.log(3)) < CLASSIFIER_INITIAL_LOSS_TOLERANCE


def test_one_step_moves_every_parameter():
    samples, _ = dealt_hand_samples()
    net = seeded_net()
    before = []
    for parameter in net.parameters():
        before.append(parameter.detach().clone())

    learner_step(net, control_optimizer(net), samples)

    for (name, parameter), old_value in zip(net.named_parameters(), before):
        assert not torch.equal(parameter, old_value), name


def test_large_gradient_is_clipped_to_the_cap():
    samples, _ = dealt_hand_samples()
    large_samples = []
    for state_vector, history, history_mask, taken_action, hand_return in samples:
        scaled_return = hand_return * LARGE_RETURN_SCALE
        large_samples.append((state_vector, history, history_mask, taken_action, scaled_return))

    net = seeded_net()
    _, norm_before_clip = learner_step(net, control_optimizer(net), large_samples)

    # The batch really did need clipping, and what is left on the weights is exactly the cap.
    assert norm_before_clip > MAX_GRAD_NORM
    assert gradient_norm(net) == pytest.approx(MAX_GRAD_NORM, abs=CLIP_TOLERANCE)


def test_old_gradients_are_cleared_before_an_update():
    samples, _ = dealt_hand_samples()

    # The reference: one update on a net with no gradients stored yet.
    clean_net = seeded_net()
    _, clean_norm = learner_step(clean_net, control_optimizer(clean_net), samples)

    # The same weights, but with a gradient already stored on every one of them, as an earlier
    # update leaves behind. backward() adds to what is stored, so if learner_step did not
    # clear it first, the gradient it measures would be this batch's plus the leftover.
    stale_net = seeded_net()
    for parameter in stale_net.parameters():
        parameter.grad = torch.full_like(parameter, STALE_GRADIENT_VALUE)
    _, stale_norm = learner_step(stale_net, control_optimizer(stale_net), samples)

    # Same weights and same batch, so the same gradient: the leftover had no effect.
    assert stale_norm == pytest.approx(clean_norm, rel=STALE_GRADIENT_TOLERANCE)


def test_optimizer_is_built_as_named():
    net = seeded_net()
    # The settings are written out here, not imported, so a changed constant fails the gate.
    rmsprop = build_optimizer(net, "rmsprop", ARM_LEARNING_RATE)
    assert type(rmsprop) is torch.optim.RMSprop
    settings = rmsprop.param_groups[0]
    assert settings["lr"] == ARM_LEARNING_RATE
    assert settings["alpha"] == 0.99
    assert settings["eps"] == 1e-5
    assert settings["momentum"] == 0.0
    assert settings["weight_decay"] == 0.0

    adam = build_optimizer(net, "adam", ARM_LEARNING_RATE)
    assert type(adam) is torch.optim.Adam
    settings = adam.param_groups[0]
    assert settings["lr"] == ARM_LEARNING_RATE
    assert settings["betas"] == (0.9, 0.999)
    assert settings["eps"] == 1e-8
    assert settings["weight_decay"] == 0.0

    # A name that is not one of the two must stop the run, not train the control.
    with pytest.raises(ValueError):
        build_optimizer(net, "adamw", ARM_LEARNING_RATE)


def test_shuffle_moves_only_the_returns():
    samples, _trace = dealt_hand_samples()
    returns_before = []
    for sample in samples:
        returns_before.append(sample[4])

    shuffled = shuffle_returns(samples, random.Random(SHUFFLE_STREAM))

    # The input list is not changed: train still holds the real batch if it wants it.
    returns_after_call = []
    for sample in samples:
        returns_after_call.append(sample[4])
    assert returns_after_call == returns_before

    # State, history, history mask and taken action stay where they were. `is` compares the
    # objects themselves, so a copy with equal numbers would not pass either.
    assert len(shuffled) == len(samples)
    for sample, shuffled_sample in zip(samples, shuffled):
        for part_index in range(4):
            assert shuffled_sample[part_index] is sample[part_index]

    # The returns are the same numbers, each as often as before, in another order. So the
    # batch's mean and variance are untouched and only the pairing with decisions is broken.
    shuffled_returns = []
    for shuffled_sample in shuffled:
        shuffled_returns.append(shuffled_sample[4])
    assert sorted(shuffled_returns) == sorted(returns_before)
    assert shuffled_returns != returns_before

    # The order comes from the rng alone: the same stream name gives the same order.
    again = shuffle_returns(samples, random.Random(SHUFFLE_STREAM))
    again_returns = []
    for again_sample in again:
        again_returns.append(again_sample[4])
    assert again_returns == shuffled_returns


@pytest.mark.parametrize("head", ["mse", HEAD_CLASSIFIER])
def test_same_seed_gives_identical_results(head):
    # catches (classifier): a classifier path that draws from an unseeded stream.
    first = train_one_process(DEV_SEED, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE, head=head)
    second = train_one_process(DEV_SEED, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE, head=head)
    other_seed = train_one_process(DEV_SEED + 1, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE, head=head)

    assert len(first) == LOOP_UPDATES
    assert first == second
    # The seed is really used: another one gives other weights and other hands.
    assert first != other_seed


def test_one_process_loop_trains_on_each_sample_once(monkeypatch):
    # Stands in for learner_step inside train_one_process: keeps each batch it is handed and
    # updates nothing. The loop's own code, which cuts the batches, is what is under test.
    batches = []

    def recording_learner_step(q_net, optimizer, samples):
        batches.append(samples)
        return 0.0, 0.0

    monkeypatch.setattr("stacker.dmc.learner_step", recording_learner_step)
    train_one_process(DEV_SEED, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE)

    assert len(batches) == LOOP_UPDATES
    # id() is the identity of the sample object itself, so a sample handed over in two batches
    # shows up as a repeated id. Every batch is still held in `batches`, so no id is reused
    # for a new object.
    sample_ids = set()
    for batch in batches:
        assert len(batch) == LOOP_SAMPLES_PER_UPDATE
        for sample in batch:
            assert id(sample) not in sample_ids
            sample_ids.add(id(sample))


def test_stream_name_uses_seed_actor_and_hand():
    name = hand_stream_name(DEV_SEED, STREAM_ACTOR, STREAM_HAND)
    # Each input alone changes the name: another run, another actor, another hand.
    assert hand_stream_name(DEV_SEED + 1, STREAM_ACTOR, STREAM_HAND) != name
    assert hand_stream_name(DEV_SEED, STREAM_ACTOR + 1, STREAM_HAND) != name
    assert hand_stream_name(DEV_SEED, STREAM_ACTOR, STREAM_HAND + 1) != name
    # And the name is a fixed function of them: the same hand is dealt on a rerun.
    assert hand_stream_name(DEV_SEED, STREAM_ACTOR, STREAM_HAND) == name


def test_training_streams_are_not_evaluation_streams():
    name = hand_stream_name(DEV_SEED, STREAM_ACTOR, STREAM_HAND)
    # The evaluation and throughput streams for the same numbers (characterize.py): one per
    # hand, and one per worker and hand. Both start with the seed's digits.
    assert name != f"{DEV_SEED}:{STREAM_HAND}"
    assert name != f"{DEV_SEED}:{STREAM_ACTOR}:{STREAM_HAND}"
    # The prefix is what keeps them apart for every seed: no evaluation stream starts with it.
    assert name.startswith("train:")


def test_actor_plays_the_hands_its_stream_names():
    # Epsilon 0 and fixed weights: a hand is then fully set by the rng it is dealt and played
    # from, so equal samples mean equal streams.
    q_net = seeded_net()

    # actor_loop sets torch to one thread for its own process. Here it runs in the test
    # process, so the setting is put back afterwards for the tests that follow.
    threads_before = torch.get_num_threads()
    hands_by_actor = []
    try:
        for actor_id in range(STREAM_GATE_ACTORS):
            sample_queue = StoppingQueue(STREAM_GATE_HANDS)
            with pytest.raises(QueueFull):
                actor_loop(actor_id, DEV_SEED, q_net, sample_queue, 0.0, HISTORY_WINDOW)

            hands = []
            for item_index in range(len(sample_queue.items)):
                queued_actor_id, hand_index, held_updates, samples = sample_queue.items[item_index]
                # Each queued item is labelled with the actor and the hand number that played
                # it; the old actor holds no copy of the weights, so it has no version.
                assert queued_actor_id == actor_id
                assert hand_index == item_index
                assert held_updates is None
                hands.append(samples)
            assert len(hands) == STREAM_GATE_HANDS
            hands_by_actor.append(hands)
    finally:
        torch.set_num_threads(threads_before)

    # The actor's n-th hand is the hand its stream name gives: seed, actor id and hand number
    # all reach hand_stream_name, and the hand number goes up by one per hand.
    for actor_id in range(STREAM_GATE_ACTORS):
        for hand_index in range(STREAM_GATE_HANDS):
            hand_rng = random.Random(hand_stream_name(DEV_SEED, actor_id, hand_index))
            state = new_hand(NUM_PLAYERS, hand_rng)
            expected_samples, _ = self_play_hand(state, q_net, 0.0, hand_rng)
            assert samples_equal(hands_by_actor[actor_id][hand_index], expected_samples)

    # And those hands really differ: two actors do not play the same deal, and one actor does
    # not play the same deal twice. Without this, training would count one hand many times.
    assert not samples_equal(hands_by_actor[0][0], hands_by_actor[1][0])
    assert not samples_equal(hands_by_actor[0][0], hands_by_actor[0][1])


def chair_hand(monkeypatch):
    """
    Plays one dev-seed hand with a fresh seeded net in every seat but CHAIR_TEST_SEAT, which
    heuristic_agent plays. A spy stands in for heuristic_agent and play_out is wrapped, so the
    test sees what the heuristic was asked and what the hand scored.
    Returns (samples, trace, chair_calls, hand_points):
      chair_calls: list in call order of (seat the observation was for, action returned).
      hand_points: the final score per seat, as play_out gave it.
    """
    chair_calls = []
    played = []

    def spy_heuristic(observation, actions, rng):
        action = heuristic_agent(observation, actions, rng)
        chair_calls.append((observation.seat, action))
        return action

    def recording_play_out(state, agents, rng):
        trace, hand_points = play_out(state, agents, rng)
        played.append(hand_points)
        return trace, hand_points

    # self_play_hand looks both names up in stacker.dmc when it runs, so these are what it uses.
    monkeypatch.setattr("stacker.dmc.heuristic_agent", spy_heuristic)
    monkeypatch.setattr("stacker.dmc.play_out", recording_play_out)
    rng = random.Random(DEV_SEED)
    state = new_hand(NUM_PLAYERS, rng)
    samples, trace = self_play_hand(
        state, seeded_net(), EPSILON, rng, HISTORY_WINDOW, CHAIR_TEST_SEAT
    )
    return samples, trace, chair_calls, played[0]


def test_chair_seat_is_played_by_the_heuristic(monkeypatch):
    # catches: the net playing the chair seat, or the heuristic sat in the wrong seat.
    samples, trace, chair_calls, hand_points = chair_hand(monkeypatch)

    # The chair seat's turns, in play order, as the trace records them.
    chair_turns = []
    for seat, action in trace:
        if seat == CHAIR_TEST_SEAT:
            chair_turns.append((seat, action))

    # The heuristic was asked once per chair turn, only ever for the chair seat, and every move
    # it gave is the move the trace shows: so it, and nothing else, played that seat.
    assert len(chair_turns) > 0
    assert chair_calls == chair_turns


def test_chair_decisions_are_not_samples(monkeypatch):
    # catches: the chair's decisions stored as samples, or a net seat's decisions dropped.
    samples, trace, chair_calls, hand_points = chair_hand(monkeypatch)

    # The net's turns: the trace with the chair seat's turns skipped.
    net_turns = []
    for seat, action in trace:
        if seat != CHAIR_TEST_SEAT:
            net_turns.append((seat, action))

    # One sample per net turn, in the same order: each stores the action that turn played and
    # carries that turn's seat's final score.
    assert len(samples) == len(net_turns)
    for (seat, action), sample in zip(net_turns, samples):
        state_vector, history, history_mask, taken_action, hand_return = sample
        assert np.array_equal(taken_action[0], encode_action(action))
        assert hand_return == float(hand_points[seat])


def test_chair_seat_is_uniform():
    # catches: a fixed chair seat, or a draw that can never pick some seat (randrange(3)).
    counts = [0] * NUM_PLAYERS
    for hand_index in range(CHAIR_DRAWS):
        # The draw comes after the deal from the same rng, as in actor_loop.
        hand_rng = random.Random(hand_stream_name(DEV_SEED, STREAM_ACTOR, hand_index))
        state = new_hand(NUM_PLAYERS, hand_rng)
        counts[draw_chair_seat(hand_rng, len(state.hands))] += 1

    for seat in range(NUM_PLAYERS):
        assert CHAIR_COUNT_LOW <= counts[seat] <= CHAIR_COUNT_HIGH, counts


def test_actor_plays_the_chair_its_stream_names():
    # catches: actor_loop ignoring heuristic_chair, or drawing the seat before the deal or from
    # another rng (either way the hand differs from the one its stream name gives).
    q_net = seeded_net()
    threads_before = torch.get_num_threads()
    sample_queue = StoppingQueue(STREAM_GATE_HANDS)
    try:
        with pytest.raises(QueueFull):
            actor_loop(0, DEV_SEED, q_net, sample_queue, 0.0, HISTORY_WINDOW, True)
    finally:
        torch.set_num_threads(threads_before)

    # Each queued hand is the hand its stream deals, with the chair seat drawn just after the
    # deal from the same rng, played at epsilon 0.
    assert len(sample_queue.items) == STREAM_GATE_HANDS
    for hand_index in range(STREAM_GATE_HANDS):
        hand_rng = random.Random(hand_stream_name(DEV_SEED, 0, hand_index))
        state = new_hand(NUM_PLAYERS, hand_rng)
        chair_seat = draw_chair_seat(hand_rng, len(state.hands))
        expected_samples, _ = self_play_hand(
            state, q_net, 0.0, hand_rng, HISTORY_WINDOW, chair_seat
        )
        queued_samples = sample_queue.items[hand_index][3]
        assert samples_equal(queued_samples, expected_samples)


def parameter_count(net):
    """The number of weights in a net: an int."""
    total = 0
    for parameter in net.parameters():
        total += parameter.numel()
    return total


def test_train_passes_the_arm_settings_on(tmp_path, monkeypatch):
    # Every argument is off its default, so one that train accepts and then ignores shows up
    # as a control value below.
    actor_arguments = []
    learner_optimizers = []
    learner_batches = []
    learner_heads = []

    # The real "spawn" context, wrapped so each actor's arguments are seen as it is started.
    real_context = torch.multiprocessing.get_context("spawn")

    class RecordingContext:
        Queue = real_context.Queue

        def Process(self, target, args, daemon):
            actor_arguments.append(args)
            return real_context.Process(target=target, args=args, daemon=daemon)

    def recording_get_context(method):
        return RecordingContext()

    def recording_learner_step(q_net, optimizer, samples):
        learner_optimizers.append(optimizer)
        learner_batches.append(samples)
        learner_heads.append(q_net.head)
        return 0.0, 0.0

    monkeypatch.setattr(torch.multiprocessing, "get_context", recording_get_context)
    monkeypatch.setattr("stacker.dmc.learner_step", recording_learner_step)
    q_net, actor_net = train(
        DEV_SEED,
        SMOKE_ACTORS,
        tmp_path,
        SMOKE_UPDATES,
        samples_per_update=SMOKE_SAMPLES_PER_UPDATE,
        checkpoint_every=SMOKE_UPDATES,
        learning_rate=ARM_LEARNING_RATE,
        epsilon=0.0,
        optimizer="adam",
        width=ARM_WIDTH,
        hidden_layers=ARM_HIDDEN_LAYERS,
        history_hidden=ARM_HISTORY_HIDDEN,
        history_window=ARM_HISTORY_WINDOW,
        heuristic_chair=True,
        head=HEAD_CLASSIFIER,
    )

    # The sizes and the head reached the learner's net and the actors' net.
    assert parameter_count(q_net) == ARM_CLASSIFIER_PARAMETER_COUNT
    assert parameter_count(actor_net) == ARM_CLASSIFIER_PARAMETER_COUNT
    # The learner was handed the classifier net at every update, so it picked the classifier loss.
    assert learner_heads == [HEAD_CLASSIFIER] * SMOKE_UPDATES

    # The learner trained with the optimizer and the rate that were named, at every update.
    assert len(learner_optimizers) == SMOKE_UPDATES
    for optimizer in learner_optimizers:
        assert type(optimizer) is torch.optim.Adam
        assert optimizer.param_groups[0]["lr"] == ARM_LEARNING_RATE

    # Every actor was started with the run's epsilon and the chair setting: actor_loop's fifth
    # and seventh arguments.
    assert len(actor_arguments) == SMOKE_ACTORS
    for arguments in actor_arguments:
        assert arguments[4] == 0.0
        assert arguments[6] is True

    # The actors encoded at the run's window: every sample the learner got has 96 history rows.
    for samples in learner_batches:
        for sample in samples:
            history = sample[1]
            assert history.shape == (ARM_HISTORY_WINDOW, MOVE_DIM)


def test_two_actors_feed_the_learner(tmp_path):
    q_net, actor_net = train(
        DEV_SEED,
        SMOKE_ACTORS,
        tmp_path,
        SMOKE_UPDATES,
        samples_per_update=SMOKE_SAMPLES_PER_UPDATE,
        checkpoint_every=SMOKE_UPDATES,
    )

    log_lines = []
    with open(tmp_path / "log.jsonl") as log_file:
        for line in log_file:
            log_lines.append(json.loads(line))
    assert len(log_lines) == SMOKE_UPDATES
    for line_index, log_line in enumerate(log_lines):
        assert math.isfinite(log_line["loss"])
        assert math.isfinite(log_line["gradient_norm"])
        # samples_total is the learning curve's x-axis: the samples trained on so far.
        assert log_line["samples_total"] == (line_index + 1) * SMOKE_SAMPLES_PER_UPDATE
        # Returns are 0, 1 or 2, so a batch's variance is at most 1 (half 0s, half 2s), and
        # above 0 unless every decision in the batch had the same return.
        assert 0.0 < log_line["return_variance"] <= MAX_RETURN_VARIANCE

    # samples_by_actor is a running total, so the last line counts the whole run: both actors
    # delivered, and together at least what the updates used.
    samples_by_actor = log_lines[-1]["samples_by_actor"]
    assert len(samples_by_actor) == SMOKE_ACTORS
    for count in samples_by_actor:
        assert count > 0
    assert sum(samples_by_actor) >= SMOKE_UPDATES * SMOKE_SAMPLES_PER_UPDATE
    # The old actors play on the shared net itself, so no lag is measured.
    assert log_lines[-1]["actor_lag_mean"] is None
    assert log_lines[-1]["actor_lag_max"] is None

    # Equation 12: after the last update the actors' net holds the learner's weights.
    learner_weights = q_net.state_dict()
    actor_weights = actor_net.state_dict()
    for name in learner_weights:
        assert torch.equal(actor_weights[name], learner_weights[name]), name

    # The last checkpoint loads into a fresh net and holds those same weights.
    restored = QNet()
    restored.load_state_dict(torch.load(tmp_path / f"checkpoint_{SMOKE_UPDATES:06d}.pt"))
    restored_weights = restored.state_dict()
    for name in learner_weights:
        assert torch.equal(restored_weights[name], learner_weights[name]), name

    # Checkpoint 0 is the untrained net: the weights QNet() draws right after the seed is set.
    # It is the point every later checkpoint is compared with, so one saved after any update,
    # or from another seed, would shift the whole learning curve.
    torch.manual_seed(DEV_SEED)
    untrained_weights = QNet().state_dict()
    saved_weights = torch.load(tmp_path / f"checkpoint_{0:06d}.pt")
    assert set(saved_weights) == set(untrained_weights)
    for name in untrained_weights:
        assert torch.equal(saved_weights[name], untrained_weights[name]), name


def test_shuffle_control_is_off_unless_asked_for(tmp_path, monkeypatch):
    # Flag off: shuffle_returns is never called, so the learner trains on the real pairing.
    shuffle_calls, learner_batches, log_lines = recorded_train(
        monkeypatch, tmp_path / "plain", False
    )
    assert shuffle_calls == []
    assert len(learner_batches) == SMOKE_UPDATES

    # The logged return variance is the variance of the returns the learner was handed,
    # dividing by the batch size (pvariance), not by one less.
    for batch, log_line in zip(learner_batches, log_lines):
        expected_variance = statistics.pvariance(returns_of(batch))
        logged_variance = log_line["return_variance"]
        assert logged_variance == pytest.approx(expected_variance, abs=VARIANCE_TOLERANCE)

    # Flag on: one shuffle per update, on a full batch, and the learner gets its result.
    shuffle_calls, learner_batches, log_lines = recorded_train(
        monkeypatch, tmp_path / "shuffled", True
    )
    assert len(shuffle_calls) == SMOKE_UPDATES
    assert len(learner_batches) == SMOKE_UPDATES
    for shuffle_call, batch, log_line in zip(shuffle_calls, learner_batches, log_lines):
        handed_in, shuffled = shuffle_call
        assert len(handed_in) == SMOKE_SAMPLES_PER_UPDATE
        # `is`: the learner's batch is the very list the shuffle returned, not the real one.
        assert batch is shuffled
        assert returns_of(batch) != returns_of(handed_in)
        # A shuffle keeps the numbers, so the logged variance is the same as the real batch's.
        expected_variance = statistics.pvariance(returns_of(handed_in))
        logged_variance = log_line["return_variance"]
        assert logged_variance == pytest.approx(expected_variance, abs=VARIANCE_TOLERANCE)


def test_used_run_directory_is_refused(tmp_path):
    # A log.jsonl marks an earlier run. train must stop before starting any actor, so this
    # test is instant: nothing is spawned.
    log_path = tmp_path / "log.jsonl"
    log_path.write_text("earlier run\n")
    with pytest.raises(FileExistsError):
        train(DEV_SEED, SMOKE_ACTORS, tmp_path, SMOKE_UPDATES)
    # The earlier log is untouched.
    assert log_path.read_text() == "earlier run\n"


def test_training_code_never_names_the_sealed_deals():
    # A text check, not a run: it fails when someone wires data/sealed-deals/ into training.
    for relative_path in TRAINING_CODE_FILES:
        source = (REPO_ROOT / relative_path).read_text()
        assert "sealed-deals" not in source, relative_path
        assert "sealed_deals" not in source, relative_path


def test_train_script_refuses_a_test_seed():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_dmc.py"),
        "--seed", str(REFUSED_SEED), "--actors", "1", "--updates", "1",
        "--run-name", REFUSED_RUN_NAME,
    ]
    finished = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
    )
    assert finished.returncode != 0
    # The refusal comes before anything is created: no run directory is left behind.
    assert not (REPO_ROOT / "runs" / "specialist_explore" / REFUSED_RUN_NAME).exists()


def test_train_script_passes_its_arm_flags_on():
    # A tiny real run through the script with every arm flag off the control, against a tiny
    # control run as its base. The arm's run name is left to the script, so the tag's place in
    # it is checked too.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    runs_dir = REPO_ROOT / "runs" / "specialist_explore"
    # Left behind by an earlier failed run of this test, the directory would be refused.
    for old_run_dir in runs_dir.glob(f"*_seed{DEV_SEED}_{SCRIPT_ARM_TAG}"):
        shutil.rmtree(old_run_dir)
    shutil.rmtree(runs_dir / SCRIPT_BASE_RUN, ignore_errors=True)
    tiny_run = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_dmc.py"),
        "--seed", str(DEV_SEED), "--actors", "1", "--updates", "1",
        "--samples-per-update", str(SMOKE_SAMPLES_PER_UPDATE),
    ]
    base_finished = subprocess.run(
        tiny_run + ["--run-name", SCRIPT_BASE_RUN],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
    )
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_dmc.py"),
        "--seed", str(DEV_SEED), "--actors", "1", "--updates", "1",
        "--samples-per-update", str(SMOKE_SAMPLES_PER_UPDATE),
        "--width", str(ARM_WIDTH), "--layers", str(ARM_HIDDEN_LAYERS),
        "--history-hidden", str(ARM_HISTORY_HIDDEN), "--lr", str(ARM_LEARNING_RATE),
        "--history-window", str(ARM_HISTORY_WINDOW),
        "--epsilon", "0.0", "--optimizer", "adam", "--heuristic-chair", "--head", "classifier",
        "--arm", SCRIPT_ARM_TAG, "--base-run", SCRIPT_BASE_RUN,
    ]
    finished = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
    )
    # Read everything the asserts need, then remove the run, so a failure leaves nothing behind.
    run_dirs = list(runs_dir.glob(f"*_seed{DEV_SEED}_{SCRIPT_ARM_TAG}"))
    config = None
    checkpoint_weights = None
    loaded_head = None
    if len(run_dirs) == 1:
        with open(run_dirs[0] / "config.json") as file:
            config = json.load(file)
        checkpoint_path = run_dirs[0] / f"checkpoint_{0:06d}.pt"
        checkpoint_weights = torch.load(checkpoint_path)
        # Evaluation rebuilds the net from the run's config; a wrong head fails to load.
        loaded_net, _ = load_checkpoint_net(checkpoint_path, config)
        loaded_head = loaded_net.head
    for run_dir in run_dirs:
        shutil.rmtree(run_dir)
    shutil.rmtree(runs_dir / SCRIPT_BASE_RUN, ignore_errors=True)

    assert base_finished.returncode == 0, base_finished.stderr
    assert finished.returncode == 0, finished.stderr
    # Exactly one run, its name ending in the tag.
    assert len(run_dirs) == 1
    assert config["arm"] == SCRIPT_ARM_TAG
    assert config["width"] == ARM_WIDTH
    assert config["hidden_layers"] == ARM_HIDDEN_LAYERS
    assert config["history_hidden"] == ARM_HISTORY_HIDDEN
    assert config["history_window"] == ARM_HISTORY_WINDOW
    assert config["learning_rate"] == ARM_LEARNING_RATE
    assert config["epsilon"] == 0.0
    assert config["optimizer"] == "adam"
    assert config["base_run"] == SCRIPT_BASE_RUN
    assert config["heuristic_chair"] is True
    assert config["head"] == HEAD_CLASSIFIER
    assert loaded_head == HEAD_CLASSIFIER
    # The net that was trained has the arm's size, not only the record of it.
    total = 0
    for weights in checkpoint_weights.values():
        total += weights.numel()
    assert total == ARM_CLASSIFIER_PARAMETER_COUNT


def test_arm_config_may_differ_from_its_base_only_where_named():
    base_config = {"seed": 20, "width": 1024, "learning_rate": 3e-4, "git_commit": "aaa"}
    # The depth arm on a re-based width-1024 net, run at a later commit. The commit differs
    # and is bookkeeping; the width differs too, which is a forgotten --width 1024.
    arm_config = {"seed": 20, "width": 512, "learning_rate": 3e-4, "git_commit": "bbb"}

    # Named: allowed. The commit is not named and must not stop the run.
    assert train_dmc.check_against_base(arm_config, base_config, {"width"}) is None

    # Not named: refused, and the message says which key.
    with pytest.raises(SystemExit) as refusal:
        train_dmc.check_against_base(arm_config, base_config, set())
    assert "width" in str(refusal.value)
    assert "git_commit" not in str(refusal.value)


# With a GPU the flag is valid and the script would start a real run, so there is nothing to
# refuse. The check is read when pytest collects the file, on the machine running the suite.
@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has a CUDA GPU")
def test_train_script_refuses_cuda_without_a_gpu():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_dmc.py"),
        "--seed", str(DEV_SEED), "--actors", "1", "--updates", "1",
        "--device", "cuda", "--run-name", NO_GPU_RUN_NAME,
    ]
    finished = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
    )
    run_dir = REPO_ROOT / "runs" / "specialist_explore" / NO_GPU_RUN_NAME
    # Read before the asserts and removed after them, so a failing run leaves nothing behind
    # to fail the next one with "run directory exists".
    run_dir_was_made = run_dir.exists()
    shutil.rmtree(run_dir, ignore_errors=True)

    assert finished.returncode != 0
    # The script's own message, not a crash from torch part-way into the run.
    assert "torch finds no CUDA GPU" in finished.stderr
    assert not run_dir_was_made


@needs_cuda
def test_gpu_update_gives_the_cpu_loss():
    samples, _trace = dealt_hand_samples()

    cpu_net = seeded_net()
    cpu_loss, _ = learner_step(cpu_net, control_optimizer(cpu_net), samples)

    # Same starting weights, moved to the GPU before the optimizer is built, as train does.
    gpu_net = seeded_net().to("cuda")
    gpu_loss, _ = learner_step(gpu_net, control_optimizer(gpu_net), samples)

    assert abs(gpu_loss - cpu_loss) <= DEVICE_LOSS_TOLERANCE
    # The update happened on the GPU: the net was not pulled back to the CPU to match the batch.
    for name, parameter in gpu_net.named_parameters():
        assert parameter.device.type == "cuda", name


@needs_cuda
def test_gpu_run_keeps_actors_and_checkpoints_on_the_cpu(tmp_path):
    q_net, actor_net = train(
        DEV_SEED,
        SMOKE_ACTORS,
        tmp_path,
        SMOKE_UPDATES,
        samples_per_update=SMOKE_SAMPLES_PER_UPDATE,
        checkpoint_every=SMOKE_UPDATES,
        device="cuda",
    )

    # Equation 13: the learner's weights are on the GPU; the actors' copy is on the CPU, still
    # in the shared memory the actor processes read, and holds the same numbers.
    learner_weights = q_net.state_dict()
    actor_weights = actor_net.state_dict()
    for name in learner_weights:
        assert learner_weights[name].device.type == "cuda", name
        assert actor_weights[name].device.type == "cpu", name
        assert actor_weights[name].is_shared(), name
        assert torch.equal(actor_weights[name], learner_weights[name].cpu()), name

    # Every checkpoint is held on the CPU, so it loads on the laptop. No map_location here:
    # torch.load then puts each tensor back where it was saved from, which is what is checked.
    for updates_done in (0, SMOKE_UPDATES):
        saved_weights = torch.load(tmp_path / f"checkpoint_{updates_done:06d}.pt")
        for name in saved_weights:
            assert saved_weights[name].device.type == "cpu", name

    # The last checkpoint holds the learner's weights. saved_weights is the last one loaded.
    for name in learner_weights:
        assert torch.equal(saved_weights[name], learner_weights[name].cpu()), name


def fast_flags(fast_actor, actor_hands=None, actor_devices=None):
    """The fast-actor part of a parsed train_dmc command line."""
    return argparse.Namespace(
        fast_actor=fast_actor, actor_hands=actor_hands, actor_devices=actor_devices
    )


def test_plain_arm_passes_against_a_base_made_before_the_fast_actor():
    # catches: a plain run writing fast_actor false, which a base with no such key would refuse.
    base_config = {"seed": 20, "actors": 24, "width": 512}
    entries, named_keys = train_dmc.fast_actor_settings(fast_flags(False))
    arm_config = {"seed": 20, "actors": 24, "width": 256}
    arm_config.update(entries)
    assert train_dmc.check_against_base(arm_config, base_config, {"width"} | named_keys) is None


def test_fast_actor_names_its_own_keys():
    # catches: --fast-actor not naming its keys (its default hands and devices refused), and
    # the guard going slack on a fast key that nothing named.
    base_config = {"seed": 20, "actors": 24}
    entries, named_keys = train_dmc.fast_actor_settings(fast_flags(True))
    fast_config = {"seed": 20, "actors": 16}
    fast_config.update(entries)
    assert fast_config["actor_hands"] == FAST_ACTOR_HANDS
    assert fast_config["actor_devices"] == ["cpu"]
    assert train_dmc.check_against_base(fast_config, base_config, {"actors"} | named_keys) is None

    with pytest.raises(SystemExit) as refusal:
        train_dmc.check_against_base(fast_config, base_config, {"actors"})
    assert "actor_hands" in str(refusal.value)


def test_actor_flags_need_fast_actor():
    # catches: --actor-hands given without --fast-actor and silently ignored by train.
    with pytest.raises(SystemExit):
        train_dmc.fast_actor_settings(fast_flags(False, actor_hands=8))


class TrainReached(Exception):
    """Raised by the stand-in train, so main stops once it has handed over its settings."""


def test_train_script_hands_its_flags_to_train(monkeypatch, tmp_path):
    # catches: the script writing a flag into config.json but handing train() the default, so
    # the run records the arm and trains the control.
    received = {}

    def recording_train(*args, **kwargs):
        received.update(kwargs)
        raise TrainReached

    # The run directory goes to a temporary folder, and train is replaced, so nothing trains.
    monkeypatch.setattr(train_dmc, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(train_dmc, "train", recording_train)
    monkeypatch.setattr(sys, "argv", [
        "train_dmc.py", "--seed", str(DEV_SEED), "--actors", "1", "--updates", "1",
        "--heuristic-chair", "--head", "classifier",
    ])
    with pytest.raises(TrainReached):
        train_dmc.main()

    assert received["heuristic_chair"] is True
    assert received["head"] == HEAD_CLASSIFIER
