"""
Gates for the M3 Deep Monte-Carlo loop, one process (stage plan specialist_explore, stage 3
`DMC loop`, pieces 2-6).

The pick: a forced turn never calls the net; epsilon 0 takes the highest Q; epsilon 1 is uniform.
The samples: every decision of a hand is stored, forced ones included; each stored action row is
the action that was played; each return is the deciding seat's own final score.
The batch: one action row per sample (K = 1), returns with one axis.
The update: the loss falls on a fixed batch; one step moves every parameter; a gradient longer
than MAX_GRAD_NORM is cut to exactly that length.
The loop: the same seed gives the same result; another seed does not.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import random

import numpy as np
import pytest
import torch
from helpers import parse_cards
from test_net import DEV_SEED, real_decisions, seeded_net

from stacker.dmc import (
    EPSILON,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_PLAYERS,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    choose_action,
    collate_samples,
    learner_step,
    self_play_hand,
    train_one_process,
)
from stacker.encode import ACTION_DIM, HISTORY_WINDOW, MOVE_DIM, STATE_DIM, encode_action
from stacker.game import HandState, new_hand

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
# Determinism test: a small loop, about a second per run.
LOOP_UPDATES = 3
LOOP_SAMPLES_PER_UPDATE = 64


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


def test_same_seed_gives_identical_results():
    first = train_one_process(DEV_SEED, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE)
    second = train_one_process(DEV_SEED, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE)
    other_seed = train_one_process(DEV_SEED + 1, LOOP_UPDATES, LOOP_SAMPLES_PER_UPDATE)

    assert len(first) == LOOP_UPDATES
    assert first == second
    # The seed is really used: another one gives other weights and other hands.
    assert first != other_seed
