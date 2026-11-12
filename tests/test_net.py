"""
Gates for the M3 control Q-net (stage plan specialist_explore, stage 2 `net + gates`).

Every gate runs on decisions from real dealt hands, because real decisions have different numbers
of legal actions and different history lengths, which is what padding and masking must handle.

Batch: collated rows equal the encode_decision rows they came from; padding is zero and masked.
Shapes: the same in all six configs; the parameter count is the spec's.
Masking: padded action rows and masked history rows cannot change a real Q.
Batch independence: a decision's Q is the same alone and inside a batch, within rounding.
Loss: an untrained net starts near mean(return^2); only the taken action's Q gets gradient; every
parameter gets gradient.
Training: the same seed gives the same losses; eval passes repeat exactly; the net can memorize
8 decisions.
"""

import random

import torch

from stacker.cards import CONFIGS
from stacker.encode import ACTION_DIM, HISTORY_WINDOW, MOVE_DIM, STATE_DIM, encode_decision
from stacker.game import new_hand, play_out
from stacker.net import HISTORY_HIDDEN, QNet, collate_decisions, q_loss

# Dev seed for dealing hands and creating nets (rule 2: seeds >= 10 outside confirm runs).
DEV_SEED = 10
# Decisions per test batch: the first ones of a hand, so some history rows are still padding.
BATCH_SIZE = 8
# Total parameters of the control net (stage plan, Mode 2 spec part 5).
SPEC_PARAMETER_COUNT = 2_323_457
# Largest allowed gap between a decision's Q alone and in a batch. A different batch shape
# changes the order of float additions, so the two differ by rounding (about 1e-8), not exactly 0.
BATCH_TOLERANCE = 1e-6
# One made-up return per decision of the test batch, covering every placement score.
FIXED_RETURNS = (2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0)
# An untrained net's Q is near 0, so its loss is near mean(return^2); this is how near.
INITIAL_LOSS_TOLERANCE = 0.1
# Steps compared in the determinism gate.
DETERMINISM_STEPS = 5
# The overfit gate: Adam is the gate's tool, not the training optimizer (stage 3 decides that).
OVERFIT_STEPS = 200
OVERFIT_LEARNING_RATE = 1e-3
OVERFIT_LOSS_CEILING = 1e-3


def real_decisions(num_players):
    """
    Plays one dev hand with random legal actions and encodes every decision of it.
    Returns a list of (state, history, mask, action_matrix) tuples from encode_decision, in play
    order: the first has no history, later ones have full windows.
    """
    decisions = []

    def recording_agent(observation, actions, rng):
        decisions.append(encode_decision(observation, actions))
        return rng.choice(actions)

    rng = random.Random(DEV_SEED)
    state = new_hand(num_players, rng)
    play_out(state, [recording_agent] * num_players, rng)
    return decisions


def fixed_batch():
    """
    The batch most gates use: the first BATCH_SIZE decisions of a 4-player hand, collated.
    Returns (decisions, batch, taken_index, returns): decisions is the uncollated list; batch is
    collate_decisions' 5-tuple; taken_index int64 [B] names each decision's last real action, so
    the taken columns differ between rows; returns float32 [B] is FIXED_RETURNS.
    """
    decisions = real_decisions(4)[:BATCH_SIZE]
    batch = collate_decisions(decisions)
    action_mask = batch[4]
    taken_index = action_mask.sum(dim=1) - 1
    returns = torch.tensor(FIXED_RETURNS)
    return decisions, batch, taken_index, returns


def seeded_net():
    """A QNet whose starting weights are fixed by DEV_SEED."""
    torch.manual_seed(DEV_SEED)
    return QNet()


def training_losses(steps):
    """
    Trains a fresh seeded net on the fixed batch for `steps` steps.
    Returns the list of loss values (floats), one per step, each measured before its update.
    """
    _, batch, taken_index, returns = fixed_batch()
    net = seeded_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=OVERFIT_LEARNING_RATE)
    losses = []
    for _ in range(steps):
        loss = q_loss(net(*batch), taken_index, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_collated_rows_equal_their_encoded_rows():
    decisions, batch, _, _ = fixed_batch()
    state, history, history_mask, actions, action_mask = batch

    for index, decision in enumerate(decisions):
        decision_state, decision_history, decision_mask, action_matrix = decision
        assert (state[index].numpy() == decision_state).all()
        assert (history[index].numpy() == decision_history).all()
        assert (history_mask[index].numpy() == decision_mask).all()

        # Real actions sit at the front, unchanged; everything after them is zero and masked.
        num_actions = len(action_matrix)
        assert (actions[index, :num_actions].numpy() == action_matrix).all()
        assert (actions[index, num_actions:] == 0).all()
        assert action_mask[index, :num_actions].all()
        assert not action_mask[index, num_actions:].any()

    # The batch really has padding, so the lines above checked something.
    assert not action_mask.all()


def test_shapes_are_the_same_in_every_config():
    net = seeded_net()
    for num_players in CONFIGS:
        decisions = real_decisions(num_players)[:BATCH_SIZE]
        max_actions = 0
        for decision in decisions:
            max_actions = max(max_actions, len(decision[3]))

        batch = collate_decisions(decisions)
        state, history, history_mask, actions, action_mask = batch
        assert state.shape == (BATCH_SIZE, STATE_DIM)
        assert history.shape == (BATCH_SIZE, HISTORY_WINDOW, MOVE_DIM)
        assert history_mask.shape == (BATCH_SIZE, HISTORY_WINDOW)
        assert actions.shape == (BATCH_SIZE, max_actions, ACTION_DIM)
        assert action_mask.shape == (BATCH_SIZE, max_actions)

        history_summary = net.summarize_history(history, history_mask)
        assert history_summary.shape == (BATCH_SIZE, HISTORY_HIDDEN)
        assert net(*batch).shape == (BATCH_SIZE, max_actions)


def test_parameter_count_matches_the_spec():
    total = 0
    for parameter in QNet().parameters():
        total += parameter.numel()
    assert total == SPEC_PARAMETER_COUNT


def test_padded_actions_cannot_change_a_real_q():
    _, batch, _, _ = fixed_batch()
    state, history, history_mask, actions, action_mask = batch
    net = seeded_net()

    with torch.no_grad():
        q_before = net(*batch)

        # Fill every padded action row with ones: no real Q may move, not even by rounding.
        padded_changed = actions.clone()
        padded_changed[~action_mask] = 1.0
        q_after = net(state, history, history_mask, padded_changed, action_mask)
        assert torch.equal(q_before, q_after)
        assert (q_before[~action_mask] == 0).all()

        # Control: the net does read real actions. Change them and the Q values move.
        real_changed = actions.clone()
        real_changed[action_mask] += 1.0
        q_real_changed = net(state, history, history_mask, real_changed, action_mask)
        assert not torch.equal(q_before, q_real_changed)


def test_masked_history_cannot_change_q():
    _, batch, _, _ = fixed_batch()
    state, history, history_mask, actions, action_mask = batch
    net = seeded_net()
    # Early decisions have padded history rows, so there is something to corrupt.
    assert not history_mask.all()

    with torch.no_grad():
        q_before = net(*batch)

        padded_changed = history.clone()
        padded_changed[~history_mask] = 1.0
        q_after = net(state, padded_changed, history_mask, actions, action_mask)
        assert torch.equal(q_before, q_after)

        # Control: the net does read real history. Change it and the Q values move.
        real_changed = history.clone()
        real_changed[history_mask] += 1.0
        q_real_changed = net(state, real_changed, history_mask, actions, action_mask)
        assert not torch.equal(q_before, q_real_changed)


def test_q_is_the_same_alone_and_in_a_batch():
    decisions, batch, _, _ = fixed_batch()
    net = seeded_net()

    with torch.no_grad():
        q_batch = net(*batch)
        for index, decision in enumerate(decisions):
            q_alone = net(*collate_decisions([decision]))[0]
            num_actions = len(q_alone)
            gap = (q_alone - q_batch[index, :num_actions]).abs().max()
            assert gap < BATCH_TOLERANCE


def test_initial_loss_is_near_mean_squared_return():
    _, batch, taken_index, returns = fixed_batch()
    net = seeded_net()

    with torch.no_grad():
        loss = q_loss(net(*batch), taken_index, returns)
    # If Q were exactly 0 the loss would be mean(return^2). A much larger or smaller start
    # means a wrong output scale or a wrong target.
    mean_squared_return = (returns**2).mean()
    assert abs(loss - mean_squared_return) < INITIAL_LOSS_TOLERANCE


def test_only_the_taken_q_gets_gradient():
    _, batch, taken_index, returns = fixed_batch()
    net = seeded_net()

    q_values = net(*batch)
    # q_values is computed, not a parameter, so torch keeps its gradient only when asked.
    q_values.retain_grad()
    q_loss(q_values, taken_index, returns).backward()

    taken_cells = torch.zeros_like(q_values, dtype=torch.bool)
    taken_cells[torch.arange(BATCH_SIZE), taken_index] = True
    assert (q_values.grad[taken_cells] != 0).all()
    assert (q_values.grad[~taken_cells] == 0).all()


def test_every_parameter_gets_gradient():
    _, batch, taken_index, returns = fixed_batch()
    net = seeded_net()
    q_loss(net(*batch), taken_index, returns).backward()

    # A parameter with no gradient is cut off from the loss and would never train.
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_same_seed_gives_identical_losses():
    assert training_losses(DETERMINISM_STEPS) == training_losses(DETERMINISM_STEPS)


def test_eval_passes_are_bit_identical():
    _, batch, _, _ = fixed_batch()
    net = seeded_net()

    with torch.no_grad():
        net.eval()
        q_first = net(*batch)
        q_second = net(*batch)
        assert torch.equal(q_first, q_second)

        # The net has no dropout or normalization layers, so train mode computes the same thing.
        net.train()
        assert torch.equal(q_first, net(*batch))


def test_net_can_memorize_eight_decisions():
    # A net that cannot drive the loss to about 0 on 8 fixed decisions has a broken path
    # somewhere between its inputs, its output and the loss.
    losses = training_losses(OVERFIT_STEPS)
    assert losses[-1] < OVERFIT_LOSS_CEILING
