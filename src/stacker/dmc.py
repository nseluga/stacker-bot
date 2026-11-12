"""
The M3 Deep Monte-Carlo training loop (architecture plan §6; spec in
docs/stage-plans/specialist_explore.md, stage 3, equations 6-12).

Actors play hands against themselves with the Q-net and store one sample per decision; a learner
fits the net so that the Q of each action played matches the score the hand really gave. The
pieces, in build order:

- The pick: choose_action selects a seat's play. Mostly it takes the action the net scores
  highest; with a small probability epsilon it takes a random one, so the net also sees plays it
  currently rates low (epsilon-greedy).
- The samples: self_play_hand plays one hand with the net in every seat and records each
  decision. A decision's training target is the score its seat got when the hand ended (the
  Monte-Carlo return), so the targets are filled in only after the hand is over.
- The batch: collate_samples stacks samples into the tensors the net and the loss take.
- The update: learner_step changes the weights once, so that the Q of each action played moves
  toward the score its hand gave. The size of one change is capped (gradient clipping).
- The loop, one process: train_one_process alternates the two, playing hands until a batch of
  samples is full and then updating. One seed replays it exactly.
"""

import random

import torch

from stacker.encode import encode_decision
from stacker.game import new_hand, play_out
from stacker.net import QNet, collate_decisions, q_loss

# Cap on the length of the whole gradient in one update. DouZero's default (--max_grad_norm 40.,
# douzero/dmc/arguments.py); decision log 2026-11-12, loop control.
MAX_GRAD_NORM = 40.0

# The control loop settings are DouZero's code defaults (douzero/dmc/arguments.py); decision log
# 2026-11-12, loop control.
# Probability of a random action on a turn with a choice (--exp_epsilon).
EPSILON = 0.01
# RMSprop: step size (--learning_rate), how much of the running average of squared gradients is
# kept each step (--alpha), and the small number added to its square root so a step never
# divides by zero (--epsilon). Momentum is 0 (--momentum).
LEARNING_RATE = 1e-4
RMSPROP_ALPHA = 0.99
RMSPROP_EPS = 1e-5
RMSPROP_MOMENTUM = 0.0

# M3 trains a specialist for one config: 4 players, 1 deck.
NUM_PLAYERS = 4


def choose_action(q_net, decision, epsilon, rng):
    """
    Picks which legal action to play for one decision (spec part 7, equations 6-7).
    q_net: a QNet. decision: (state, history, mask, action_matrix) from encode_decision;
      action_matrix is [K, ACTION_DIM], one row per legal action, in legal_actions order.
    epsilon: probability of a random pick on a turn that has a choice.
    rng: a random.Random; the only source of randomness here.
    Returns an int in [0, K): the row of the chosen action.
    """
    action_matrix = decision[3]
    num_actions = len(action_matrix)

    # Equation 6: a forced turn has nothing to choose, so the net is not called and no random
    # number is drawn.
    if num_actions == 1:
        return 0

    # Equation 7, exploring: every legal action is equally likely.
    if rng.random() < epsilon:
        return rng.randrange(num_actions)

    # Equation 7, greedy: score the K actions as a batch of one decision. With one decision in
    # the batch there are no padded action rows, so the argmax needs no action mask.
    batch = collate_decisions([decision])
    # no_grad: acting never trains, so torch keeps no record for gradients.
    with torch.no_grad():
        q_values = q_net(*batch)  # [1, K]

    # argmax returns the lowest index among equal scores, so ties resolve the same way every run.
    return int(q_values[0].argmax())


def self_play_hand(state, q_net, epsilon, rng):
    """
    Plays one dealt hand with q_net in every seat and returns a training sample per decision
    (spec part 8, equations 8-9).
    state: a HandState, already dealt (changed in place, as play_out does).
    q_net: a QNet. epsilon: exploration probability, passed to choose_action.
    rng: a random.Random; drives every random pick in the hand.
    Returns (samples, trace).
      samples: list in play order, one per decision, forced turns included. Each is
        (state_vector [STATE_DIM], history [H, MOVE_DIM], history_mask [H],
        taken_action [1, ACTION_DIM], hand_return): the first three as encode_decision gives
        them, taken_action the one row of the action that was played, hand_return a float, the
        deciding seat's final score (2, 1 or 0).
      trace: the list of (seat, action) from play_out; samples[n] is the decision behind trace[n].
    """
    # Filled by the agent below while the hand is played. The return is not known yet, so each
    # record keeps the seat, to look its score up afterwards.
    records = []

    def recording_agent(observation, actions, agent_rng):
        decision = encode_decision(observation, actions)
        index = choose_action(q_net, decision, epsilon, agent_rng)
        state_vector, history, history_mask, action_matrix = decision

        # Only the played action is kept: the loss gives the other K - 1 actions no gradient
        # (equation 5). The slice keeps a leading axis, [1, ACTION_DIM], so the sample still
        # looks like a decision with one candidate. copy() lets the K-row matrix be freed.
        taken_action = action_matrix[index : index + 1].copy()
        records.append((observation.seat, state_vector, history, history_mask, taken_action))
        return actions[index]

    # Self-play: the same net, and so the same agent, sits in every seat.
    agents = []
    for _ in state.hands:
        agents.append(recording_agent)
    trace, hand_points = play_out(state, agents, rng)

    # Equation 8: every decision a seat made gets that seat's final score as its target. There
    # is no discount and no reward before the end, so early and late decisions get the same one.
    samples = []
    for seat, state_vector, history, history_mask, taken_action in records:
        hand_return = float(hand_points[seat])
        samples.append((state_vector, history, history_mask, taken_action, hand_return))
    return samples, trace


def collate_samples(samples):
    """
    Stacks B samples into the tensors QNet.forward and q_loss take (spec part 8, equation 9).
    samples: list of B samples from self_play_hand, each (state_vector, history, history_mask,
      taken_action [1, ACTION_DIM], hand_return).
    Returns (net_inputs, taken_index, returns).
      net_inputs: (state [B, STATE_DIM], history [B, H, MOVE_DIM], history_mask [B, H],
        actions [B, 1, ACTION_DIM], action_mask [B, 1]), in QNet.forward's argument order. Each
        sample has one action row, so K = 1, nothing is padded and action_mask is all True.
      taken_index: int64 [B], all 0. returns: float32 [B].
    """
    # A sample without its return has the same four parts as an encode_decision result with one
    # legal action, so collate_decisions can stack it unchanged.
    decisions = []
    hand_returns = []
    for state_vector, history, history_mask, taken_action, hand_return in samples:
        decisions.append((state_vector, history, history_mask, taken_action))
        hand_returns.append(hand_return)
    net_inputs = collate_decisions(decisions)

    # The played action is the only row a sample keeps, so its column is always 0.
    taken_index = torch.zeros(len(samples), dtype=torch.int64)

    # One axis, [B], to match the taken Q in q_loss. A [B, 1] here would broadcast against [B]
    # into a [B, B] difference and average every sample against every return.
    returns = torch.tensor(hand_returns, dtype=torch.float32)
    return net_inputs, taken_index, returns


def learner_step(q_net, optimizer, samples):
    """
    Updates q_net once from one batch of samples (spec part 9, equations 10-11).
    q_net: a QNet; its weights are changed in place.
    optimizer: a torch optimizer built over q_net.parameters(). Passed in, so the caller picks
      the optimizer and its settings.
    samples: list of B samples from self_play_hand.
    Returns (loss, gradient_norm), both floats: the mean squared error on this batch before the
      update, and the length of the gradient before clipping.
    """
    net_inputs, taken_index, returns = collate_samples(samples)

    # backward() adds to the gradients already stored on the weights, so the ones from the last
    # update are cleared first. Clearing here, not after the step, leaves this update's
    # gradients readable once the function returns.
    optimizer.zero_grad()

    # Equation 10: Q of each played action against its hand's return.
    q_values = q_net(*net_inputs)  # [B, 1]
    loss = q_loss(q_values, taken_index, returns)  # scalar
    loss.backward()

    # Equation 11: treat all the weights' gradients as one long vector. If it is longer than
    # MAX_GRAD_NORM, every gradient is scaled by the same factor so the length is exactly
    # MAX_GRAD_NORM: the direction is kept and one odd batch cannot cause a huge step. The call
    # returns the length it measured before scaling.
    gradient_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    # detach(): the loss is only reported from here on, so it is cut loose from the gradient record.
    return float(loss.detach()), float(gradient_norm)


def train_one_process(seed, num_updates, samples_per_update):
    """
    Runs the whole training loop in this process: play hands, fill a batch, update, repeat
    (spec parts 7-9). The same seed gives the same result on every run.
    seed: int; fixes the net's starting weights and every hand.
    num_updates: how many learner_step calls to make.
    samples_per_update: batch size B of each update.
    Returns a list of num_updates (loss, gradient_norm) pairs, in update order, as learner_step
      gives them.
    """
    # QNet() draws its starting weights from torch's global random stream, so seeding that
    # stream first fixes them.
    torch.manual_seed(seed)
    q_net = QNet()
    optimizer = torch.optim.RMSprop(
        q_net.parameters(),
        lr=LEARNING_RATE,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        momentum=RMSPROP_MOMENTUM,
    )

    # Samples played but not yet trained on. A hand rarely ends exactly on a batch boundary, so
    # what is left over after a batch is cut starts the next one: every sample is used once.
    waiting_samples = []
    hands_played = 0
    results = []
    for _ in range(num_updates):
        while len(waiting_samples) < samples_per_update:
            # One rng per hand, named by the seed and the hand's number. The 0 is the actor id:
            # this process plays the part of actor 0 of the many-actor loop. The "train:" prefix
            # keeps the string different from every evaluation stream.
            hand_rng = random.Random(f"train:{seed}:0:{hands_played}")
            state = new_hand(NUM_PLAYERS, hand_rng)
            # The net that acts is the net being trained, so the actor always plays with the
            # newest weights (equation 12 with nothing to copy).
            samples, _ = self_play_hand(state, q_net, EPSILON, hand_rng)
            waiting_samples.extend(samples)
            hands_played += 1

        batch = waiting_samples[:samples_per_update]
        waiting_samples = waiting_samples[samples_per_update:]
        results.append(learner_step(q_net, optimizer, batch))
    return results
