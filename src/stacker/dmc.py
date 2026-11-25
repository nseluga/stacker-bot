"""
The M3 Deep Monte-Carlo training loop (architecture plan §6; spec in
docs/stage-plans/specialist_explore.md, stage 3, equations 6-12).

Actors play hands against themselves with the Q-net and store one sample per decision; a learner
fits the net so that the Q of each action played matches the score the hand really gave. The
pieces, in build order:

- The pick: choose_action selects a seat's play. Mostly it takes the action the net scores
  highest; with a small probability epsilon it takes a random one, so the net also sees plays it
  currently rates low (epsilon-greedy).
- The batched pick: batched_choices scores many decisions in one net call and takes each
  one's best legal action; choose_actions_batched adds the exploring picks, drawing from each
  hand's rng exactly as choose_action does, so the fast actor plays the same hands.
- The samples: self_play_hand plays one hand with the net in every seat and records each
  decision. A decision's training target is the score its seat got when the hand ended (the
  Monte-Carlo return), so the targets are filled in only after the hand is over.
- The lockstep hands: advance_slots moves every hand an actor holds (a HandSlot each) one
  decision with one choose_actions_batched call, and returns the samples of the hands that end.
- The batch: collate_samples stacks samples into the tensors the net and the loss take.
- The update: learner_step changes the weights once, so that the Q of each action played moves
  toward the score its hand gave. The size of one change is capped (gradient clipping).
- The loop, one process: train_one_process alternates the two, playing hands until a batch of
  samples is full and then updating. One seed replays it exactly.
- The loop, many processes: actor_loop is what one actor process runs. It plays hands with a
  net it shares with the learner and sends each hand's samples to the learner through a queue.
  train starts the actors, cuts batches from the queue, updates, and after every update copies
  the new weights into the shared net. It writes a log line per update and saves checkpoints.
- The fast actor: fast_actor_loop is what one fast actor process runs. It keeps many hands in
  play on its own net (copied from the learner's every update) and queues each finished hand.
- The weight copy for actors with their own net (the fast actor): make_shared_nets builds one
  net per GPU that the learner writes into with write_weights after each update; an actor
  copies the one on its GPU into its own net with refresh_private_net, and a version counter
  tells it whether the copy it made is whole.
"""

import copy
import json
import random
import time
from pathlib import Path

import torch

from stacker.encode import HISTORY_WINDOW, encode_decision
from stacker.game import apply_action, legal_actions, new_hand, observe, play_out, points
from stacker.heuristic import heuristic_agent
from stacker.net import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HISTORY_HIDDEN,
    DEFAULT_MLP_WIDTH,
    QNet,
    collate_decisions,
    q_loss,
)

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
# The control's optimizer, by the name build_optimizer takes.
CONTROL_OPTIMIZER = "rmsprop"
# Adam, the optimizer arm (decision log 2026-11-22, Arms: Adam): PyTorch's defaults. How much of
# the running averages of the gradient and of its square is kept each step, and the small number
# added to the square root so a step never divides by zero. No weight decay.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

# M3 trains a specialist for one config: 4 players, 1 deck.
NUM_PLAYERS = 4

# Samples in one update: DouZero's --batch_size 32 times --unroll_length 100.
SAMPLES_PER_UPDATE = 3200
# Finished hands the queue holds before actors must wait. At about 68 samples a hand this is
# about 1.4 updates' worth, close to DouZero's 50 buffers of 100 steps (1.5), so queued samples
# are at most a couple of updates older than the weights. About 64 MB when full.
QUEUE_MAX_HANDS = 64
# Longest the learner waits for one hand before the run fails. A hand takes well under a second,
# so a wait this long means the actors have died, and stopping beats hanging without a word.
QUEUE_TIMEOUT_SECONDS = 60
# Updates between saved checkpoints. A placeholder: stage 4 sets it from the probe's updates
# per hour.
CHECKPOINT_EVERY_UPDATES = 100

# The fast actor (decision log 2026-11-25, design C): hands each actor keeps in play at once,
# whose choice turns share one net call per pass, and how many learner updates it may fall
# behind before copying the new weights (1: it takes every update).
FAST_ACTOR_HANDS = 64
FAST_ACTOR_REFRESH_EVERY = 1
# The updates a fast actor's own net "holds" before its first copy: one below any real count,
# so the first refresh always makes a whole, version-checked copy (see fast_actor_loop).
HELD_BEFORE_FIRST_COPY = -1


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


def batched_choices(net, decisions, device):
    """
    Picks the best legal action for B decisions with one net call: the step a batched actor
    takes in place of B calls to choose_action (epsilon 0). The greedy half of
    choose_actions_batched; the throughput probes call it directly.
    net: a QNet already on `device`. decisions: list of B encode_decision tuples, any mix of K.
    device: a torch.device.
    Returns a list of B ints; entry b is the chosen row of decision b's action_matrix.
    """
    # collate_decisions pads every action list to the longest K in the batch.
    net_inputs = collate_decisions(decisions)
    on_device = []
    for tensor in net_inputs:
        on_device.append(tensor.to(device))
    action_mask = on_device[4]  # [B, K], True for real actions
    with torch.no_grad():
        q_values = net(*on_device)  # [B, K], padded slots hold 0
    # A padded slot holds 0, which would beat real actions that all score below 0, so it is set
    # to minus infinity before the argmax. argmax runs along the action axis, one per decision.
    masked_q = q_values.masked_fill(~action_mask, float("-inf"))
    # tolist() copies the answers to the CPU, which waits for the GPU to finish: the timing
    # around this call then covers the whole batch, not just queueing it.
    return masked_q.argmax(dim=1).tolist()


def choose_actions_batched(q_net, decisions, epsilon, rngs, device):
    """
    Picks the action for many decisions at once, each exactly as choose_action would (spec
    part 7, equations 6-7), with one net call for all the greedy picks: the fast actor's pick.
    q_net: a QNet already on `device`.
    decisions: list of N (state, history, mask, action_matrix) tuples from encode_decision, any
      mix of action counts K, forced turns (K = 1) included.
    epsilon: probability of a random pick on a turn that has a choice.
    rngs: list of N random.Random; rngs[n] is decision n's hand's rng. Several decisions may
      share one rng; each draws from it in list order.
    device: a torch.device, where q_net is.
    Returns a list of N ints; entry n is the chosen row of decision n's action_matrix.

    Each decision draws from its rng exactly what choose_action draws, in the same order:
    nothing on a forced turn, one number to decide whether to explore, and one more to pick the
    random action if it does. So a hand's rng is left where choose_action would leave it, and
    the hand's later picks and the next deal from it are unchanged.
    """
    choices = []
    # Decisions that take the net's best action, by position in `decisions`; scored together.
    greedy_positions = []
    greedy_decisions = []
    for position in range(len(decisions)):
        decision = decisions[position]
        rng = rngs[position]
        num_actions = len(decision[3])

        # Equation 6: a forced turn has nothing to choose; no net call, no random draw.
        if num_actions == 1:
            choices.append(0)
        # Equation 7, exploring: every legal action is equally likely.
        elif rng.random() < epsilon:
            choices.append(rng.randrange(num_actions))
        # Equation 7, greedy: a placeholder until the one net call below fills it in.
        else:
            choices.append(None)
            greedy_positions.append(position)
            greedy_decisions.append(decision)

    # One batched call for every greedy pick; answer i belongs to greedy_positions[i].
    if len(greedy_decisions) > 0:
        greedy_choices = batched_choices(q_net, greedy_decisions, device)
        for index in range(len(greedy_positions)):
            choices[greedy_positions[index]] = greedy_choices[index]
    return choices


def self_play_hand(state, q_net, epsilon, rng, history_window=HISTORY_WINDOW, chair_seat=None):
    """
    Plays one dealt hand with q_net in every seat and returns a training sample per decision
    (spec part 8, equations 8-9). With chair_seat set, that one seat is played by
    heuristic_agent instead (the heuristic-chair arm) and its decisions are not samples.
    state: a HandState, already dealt (changed in place, as play_out does).
    q_net: a QNet. epsilon: exploration probability, passed to choose_action.
    rng: a random.Random; drives every random pick in the hand.
    history_window: int, moves of history each decision is encoded with (H below).
    chair_seat: None (every seat is the net: the control), or an int seat in
      range(len(state.hands)) played by heuristic_agent.
    Returns (samples, trace).
      samples: list in play order, one per net decision, forced turns included. Each is
        (state_vector [STATE_DIM], history [H, MOVE_DIM], history_mask [H],
        taken_action [1, ACTION_DIM], hand_return): the first three as encode_decision gives
        them, taken_action the one row of the action that was played, hand_return a float, the
        deciding seat's final score (2, 1 or 0).
      trace: the list of (seat, action) from play_out. With no chair, samples[n] is the decision
        behind trace[n]; with a chair, the samples follow the trace with the chair's turns skipped.
    """
    # A seat outside the table would match no seat and silently give pure self-play.
    if chair_seat is not None and chair_seat not in range(len(state.hands)):
        last_seat = len(state.hands) - 1
        raise ValueError(f"chair_seat must be None or a seat 0-{last_seat}, got {chair_seat}")

    # Filled by the agent below while the hand is played. The return is not known yet, so each
    # record keeps the seat, to look its score up afterwards.
    records = []

    def recording_agent(observation, actions, agent_rng):
        decision = encode_decision(observation, actions, history_window)
        index = choose_action(q_net, decision, epsilon, agent_rng)
        state_vector, history, history_mask, action_matrix = decision

        # Only the played action is kept: the loss gives the other K - 1 actions no gradient
        # (equation 5). The slice keeps a leading axis, [1, ACTION_DIM], so the sample still
        # looks like a decision with one candidate. copy() lets the K-row matrix be freed.
        taken_action = action_matrix[index : index + 1].copy()
        records.append((observation.seat, state_vector, history, history_mask, taken_action))
        return actions[index]

    # Self-play: the same net, and so the same agent, sits in every seat but the chair.
    # The chair's decisions are not recorded. A decision's target is its seat's final score,
    # which here measures the heuristic playing on, not the net; learning from it would pull
    # the net's Q toward the heuristic's values (off-policy Monte Carlo, Sutton & Barto 5.5).
    # The heuristic draws from the same hand rng play_out hands every agent, so the hand
    # still replays exactly from its stream name.
    agents = []
    for seat in range(len(state.hands)):
        if seat == chair_seat:
            agents.append(heuristic_agent)
        else:
            agents.append(recording_agent)
    trace, hand_points = play_out(state, agents, rng)

    return samples_from_records(records, hand_points), trace


def samples_from_records(records, hand_points):
    """
    Turns one finished hand's decision records into training samples (spec part 8, equation 8).
    records: list in play order of (seat, state_vector, history, history_mask, taken_action).
    hand_points: list, the hand's final score per seat, as play_out or points gives it.
    Returns a list of (state_vector, history, history_mask, taken_action, hand_return), one per
      record, in the same order.
    """
    # Equation 8: every decision a seat made gets that seat's final score as its target. There
    # is no discount and no reward before the end, so early and late decisions get the same one.
    samples = []
    for seat, state_vector, history, history_mask, taken_action in records:
        hand_return = float(hand_points[seat])
        samples.append((state_vector, history, history_mask, taken_action, hand_return))
    return samples


class HandSlot:
    """
    One hand in play in a fast actor: the actor keeps many slots and moves them all one decision
    per pass (advance_slots), so their net calls can share one batch.
    hand_index: int, which hand of the actor's stream this is (see hand_stream_name).
    rng: the hand's random.Random; it dealt the hand and draws every pick in it.
    state: the HandState, already dealt; changed in place as the hand is played.
    trace: list of (seat, action) played so far, as play_out keeps it.
    records: one (seat, state_vector, history, history_mask, taken_action) per decision so far,
      as self_play_hand keeps them; the returns are added when the hand ends.
    """

    def __init__(self, hand_index, rng, state):
        self.hand_index = hand_index
        self.rng = rng
        self.state = state
        self.trace = []
        self.records = []


def advance_slots(slots, q_net, epsilon, history_window, device):
    """
    One pass of a fast actor (spec parts 7-8): every slot whose hand is still in play makes one
    decision, and all those decisions share one net call (choose_actions_batched).
    Each hand is played exactly as self_play_hand plays it, a turn at a time: play_out's loop
    run across many hands in step instead of one hand to the end.
    slots: list of HandSlot; changed in place.
    q_net: a QNet on `device`. epsilon, history_window: as in self_play_hand.
    device: a torch.device, where q_net is.
    Returns a list of (slot, samples), one per hand that ended in this pass, in slot order;
      samples as self_play_hand gives them. A finished slot stays in `slots`, finished; the
      caller replaces it.
    """
    # The turn each in-play slot takes this pass, gathered first so one call can pick them all.
    playing_slots = []
    pass_actions = []
    pass_decisions = []
    pass_rngs = []
    for slot in slots:
        # A finished hand has no seat to act; it waits for the caller to replace it.
        if slot.state.turn is None:
            continue
        # The same three steps play_out takes each turn: the legal actions, what the acting
        # seat can see, and that view encoded for the net.
        actions = legal_actions(slot.state)
        observation = observe(slot.state, slot.trace)
        decision = encode_decision(observation, actions, history_window)
        playing_slots.append(slot)
        pass_actions.append(actions)
        pass_decisions.append(decision)
        pass_rngs.append(slot.rng)

    choices = choose_actions_batched(q_net, pass_decisions, epsilon, pass_rngs, device)

    finished = []
    for index in range(len(playing_slots)):
        slot = playing_slots[index]
        choice = choices[index]
        state_vector, history, history_mask, action_matrix = pass_decisions[index]
        action = pass_actions[index][choice]

        # Recorded as self_play_hand records it: only the played action's row, with a leading
        # axis, and the seat, to look its final score up when the hand ends.
        taken_action = action_matrix[choice : choice + 1].copy()
        slot.records.append(
            (slot.state.turn, state_vector, history, history_mask, taken_action)
        )
        # Then the turn is played, as in play_out: the trace first, then the state.
        slot.trace.append((slot.state.turn, action))
        apply_action(slot.state, action)

        if slot.state.turn is None:
            finished.append((slot, samples_from_records(slot.records, points(slot.state))))
    return finished


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


def shuffle_returns(samples, rng):
    """
    The label-shuffle control (spec part 13, equation 14): hands the batch's returns out again
    in a random order, so no sample's return has anything to do with its decision.
    samples: list of B samples from self_play_hand. Not changed.
    rng: a random.Random; decides the order.
    Returns a new list of B samples in the same order. Each keeps its state, history, mask and
      taken action and gets the return of a random sample of the batch. The B returns are the
      same numbers as before, each used once.

    A net trained on these batches has nothing to learn from: its loss must stay at the
    variance of the batch's returns. If it goes lower, something other than the decision is
    telling the net its return.
    """
    shuffled = []
    for _, _, _, _, hand_return in samples:
        shuffled.append(hand_return)
    # In place; every order of the B returns is equally likely.
    rng.shuffle(shuffled)

    # A sample is a tuple and cannot be changed, so each one is built again with its new return.
    shuffled_samples = []
    for index in range(len(samples)):
        state_vector, history, history_mask, taken_action, _ = samples[index]
        shuffled_samples.append(
            (state_vector, history, history_mask, taken_action, shuffled[index])
        )
    return shuffled_samples


def learner_step(q_net, optimizer, samples):
    """
    Updates q_net once from one batch of samples (spec part 9, equations 10-11).
    q_net: a QNet; its weights are changed in place.
    optimizer: a torch optimizer built over q_net.parameters(). Passed in, so the caller picks
      the optimizer and its settings.
    samples: list of B samples from self_play_hand.
    Returns (loss, gradient_norm), both floats: the mean squared error on this batch before the
      update, and the length of the gradient before clipping.
    The update runs on the device q_net's weights are on (spec part 12, equation 13).
    """
    net_inputs, taken_index, returns = collate_samples(samples)

    # The batch is built on the CPU. PyTorch cannot combine tensors on different devices, so
    # every tensor of the batch is moved to where the weights are. When that is the CPU, .to()
    # hands back the same tensor and nothing is copied.
    device = next(q_net.parameters()).device
    inputs_on_device = []
    for tensor in net_inputs:
        inputs_on_device.append(tensor.to(device))
    net_inputs = inputs_on_device
    taken_index = taken_index.to(device)
    returns = returns.to(device)

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


def build_optimizer(q_net, optimizer=CONTROL_OPTIMIZER, learning_rate=LEARNING_RATE):
    """
    Builds the optimizer a run names, over q_net's weights (spec part 17).
    q_net: a QNet, already on the device it trains on: the optimizer keeps running averages per
      weight and creates them where the weights are.
    optimizer: "rmsprop" (the control) or "adam" (the optimizer arm).
    learning_rate: the step size, for either optimizer.
    Returns a torch optimizer. Every setting but the rate is a constant of this file.
    Raises ValueError for any other name, so a typo cannot quietly train the control.
    """
    if optimizer == "rmsprop":
        return torch.optim.RMSprop(
            q_net.parameters(),
            lr=learning_rate,
            alpha=RMSPROP_ALPHA,
            eps=RMSPROP_EPS,
            momentum=RMSPROP_MOMENTUM,
        )
    if optimizer == "adam":
        return torch.optim.Adam(
            q_net.parameters(), lr=learning_rate, betas=ADAM_BETAS, eps=ADAM_EPS
        )
    raise ValueError(f'optimizer must be "rmsprop" or "adam", got {optimizer!r}')


def hand_stream_name(seed, actor_id, hand_index):
    """
    The name of the random stream one training hand is dealt and played from.
    seed: the run's seed. actor_id: which actor plays the hand (0 in the one-process loop).
    hand_index: that actor's hand count, from 0.
    Returns a string; random.Random(that string) is the hand's rng.

    All three numbers are in the name, so no two runs, no two actors of a run, and no two hands
    of an actor share a deal. The "train:" prefix keeps the name different from every evaluation
    stream, which are f"{seed}:{hand}" and f"{seed}:{worker}:{hand}" (characterize.py).
    """
    return f"train:{seed}:{actor_id}:{hand_index}"


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
            # One rng per hand. The 0 is the actor id: this process plays the part of actor 0
            # of the many-actor loop.
            hand_rng = random.Random(hand_stream_name(seed, 0, hands_played))
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


def make_shared_nets(context, device_names, start_net):
    """
    Builds the nets the learner writes its weights into: one per device, readable by every
    process the parent starts (weight-copy probe, step 13f, design C; the fast actor).

    A CUDA tensor passed to a spawned process travels as a handle to the same GPU memory, not
    as a copy, so a child reading the net sees the learner's later writes. That handle holds
    only while the process that made the tensor is alive: the parent makes these nets and
    outlives every child. A CPU net needs share_memory() for the same effect.

    context: the spawn multiprocessing context the run's processes come from.
    device_names: list of str, one per net, e.g. ["cuda:0", "cuda:1"], or ["cpu", "cpu"] in
        the laptop tests.
    start_net: a QNet; each shared net is a copy of it, so it has the same sizes and weights.
        A copy, not the net itself: start_net (the learner's net, say) can move or change
        without touching the shared nets.
    Returns (shared_nets, version):
      shared_nets: list of QNet in eval mode, one per device name, holding start_net's weights.
      version: a shared integer, 0 now; the learner raises it by one before and one after each
        write (see write_weights), so a reader can tell new weights from old.
    """
    shared_nets = []
    for device_name in device_names:
        net = copy.deepcopy(start_net).to(torch.device(device_name))
        net.eval()
        if device_name == "cpu":
            net.share_memory()
        shared_nets.append(net)

    # "q": a signed 64-bit integer; the Value's own lock lets the learner raise it safely.
    version = context.Value("q", 0)
    return shared_nets, version


def write_weights(learner_net, shared_nets, version):
    """
    Copies the learner's weights into every shared net and times each copy (step 13f; the
    learner's side of the fast actor).

    The version works as a write flag as well as a counter: it is raised once before the copies
    (odd: a write is under way) and once after they have finished (even: every shared net holds
    one whole set of weights). A reader that sees the same even value before and after its own
    read knows no write overlapped it. So updates written = version // 2.

    learner_net: the learner's QNet, on its own device.
    shared_nets: list of QNet from make_shared_nets.
    version: the shared integer from make_shared_nets; even on entry.
    Returns a list of float, the seconds each shared net's copy took, in shared_nets order.
    """
    with version.get_lock():
        version.value += 1

    copy_seconds = []
    learner_weights = learner_net.state_dict()
    for shared_net in shared_nets:
        copy_start = time.perf_counter()
        # load_state_dict copies into the shared net's existing tensors, so the memory every
        # reader holds a handle to is overwritten in place; it never swaps in new tensors.
        shared_net.load_state_dict(learner_weights)
        # A GPU copy only queues work; waiting for it makes the time real and makes sure the
        # weights are in place before the version below says so.
        if shared_net_device(shared_net).type == "cuda":
            torch.cuda.synchronize(shared_net_device(shared_net))
        copy_seconds.append(time.perf_counter() - copy_start)

    with version.get_lock():
        version.value += 1
    return copy_seconds


def refresh_private_net(private_net, shared_net, version, held_updates, refresh_every):
    """
    Design C's actor side (step 13f; the fast actor's refresh): copies the shared net into the actor's own net once the
    learner is refresh_every or more updates ahead of it; otherwise does nothing.

    The copy is kept only if the version was the same even value before and after it (see
    write_weights): odd means a write was under way, and a changed value means one started
    during the copy, so either way the copy may mix old and new weights and is made again.

    private_net: the actor's own QNet, on the same device as shared_net.
    shared_net: the QNet from make_shared_nets for that device.
    version: the shared integer from make_shared_nets.
    held_updates: int, how many learner updates private_net's weights include.
    refresh_every: int >= 1, how many updates the learner may get ahead before a copy.
    Returns (held_updates, copy_seconds): the updates private_net now includes, and the
    seconds spent copying, 0.0 when no copy was due.
    """
    copy_seconds = 0.0
    while True:
        before = version.value
        # Updates fully written so far; an odd value means the next one is being written.
        if before // 2 - held_updates < refresh_every:
            return held_updates, copy_seconds
        if before % 2 == 1:
            continue

        copy_start = time.perf_counter()
        private_net.load_state_dict(shared_net.state_dict())
        # Wait for the copy to finish before reading the version again, or a write that started
        # after the copy was queued could go unseen.
        if shared_net_device(shared_net).type == "cuda":
            torch.cuda.synchronize(shared_net_device(shared_net))
        copy_seconds += time.perf_counter() - copy_start

        if version.value == before:
            return before // 2, copy_seconds


def shared_net_device(net):
    """The device a net's weights live on."""
    return next(net.parameters()).device


def actor_loop(actor_id, seed, actor_net, sample_queue, epsilon, history_window):
    """
    Plays hands forever and queues their samples: the target function of one actor process
    (spec parts 7-8). Never returns; the process that started it ends it.
    actor_id: int, this actor's number, from 0. seed: int, the run's seed.
    actor_net: a QNet whose weights are in shared memory. The learner writes new weights into
      that memory, so this actor's next forward pass uses them without being told.
    sample_queue: a bounded multiprocessing queue. Each item put on it is
      (actor_id, hand_index, None, samples): the samples of one finished hand, as
      self_play_hand gives them, and its hand number. The None stands where a fast actor puts
      the weight version it held (fast_actor_loop): this actor plays on the shared net itself,
      so it holds no copy that could be behind.
    epsilon: exploration probability, passed to self_play_hand.
    history_window: int, the run's history window, passed to self_play_hand. The samples arrive
      at the learner already encoded at this window, so the learner needs no window of its own.
    """
    # By default torch spreads one forward pass over every core. With one actor process per
    # core that would be many times more threads than cores, each waiting on the others.
    torch.set_num_threads(1)

    hands_played = 0
    while True:
        # One rng per hand: no two actors of a run, and no two runs, play the same deal.
        hand_rng = random.Random(hand_stream_name(seed, actor_id, hands_played))
        state = new_hand(NUM_PLAYERS, hand_rng)
        # The learner may be writing new weights while this hand is played, with no lock, as
        # in DouZero: a decision can be scored by a mix of the old and new weights, which
        # differ by one small update.
        samples, _ = self_play_hand(state, actor_net, epsilon, hand_rng, history_window)

        # put waits while the queue is full. That wait is the brake: actors cannot make
        # samples faster than the learner uses them and fill memory.
        # ponytail: the queue copies each hand's samples (about 1 MB) to the learner process.
        # DouZero avoids the copy with shared-memory buffers; switch if the throughput probe
        # shows the queue is the bottleneck.
        sample_queue.put((actor_id, hands_played, None, samples))
        hands_played += 1


def fast_actor_loop(
    actor_id,
    seed,
    device_name,
    shared_net,
    version,
    sample_queue,
    epsilon,
    history_window,
    hands=FAST_ACTOR_HANDS,
):
    """
    Plays hands forever and queues their samples: the target function of one fast actor
    process (spec parts 7-8; design C, decision log 2026-11-25). Never returns; the process
    that started it ends it.

    Unlike actor_loop, the actor scores with its own net on its own device, keeps `hands` hands
    in play at once and moves them all one decision per pass (advance_slots), so the pass's
    choice turns share one net call. Before each pass it copies the learner's newest weights
    from shared_net if it is behind (refresh_private_net).

    actor_id: int, from 0. seed: int, the run's seed; hand n of this actor is dealt from
      hand_stream_name(seed, actor_id, n), as actor_loop deals it.
    device_name: str, where this actor's net lives and scores, e.g. "cuda:1" or "cpu".
    shared_net, version: from make_shared_nets, the net on this actor's device that the
      learner writes into, and the shared version counter.
    sample_queue: a bounded multiprocessing queue. Each item put on it is
      (actor_id, hand_index, held_updates, samples): one finished hand's samples as
      self_play_hand gives them, its hand number, and how many learner updates the weights
      that played its last decision include.
    epsilon, history_window: as in actor_loop.
    hands: int, hands kept in play at once.
    """
    # One thread per actor, as in actor_loop: many actors share the cores.
    torch.set_num_threads(1)
    device = torch.device(device_name)

    # The actor's own net: the same sizes as the shared one, on this actor's device. Its weights
    # are replaced by the first refresh below, so the copy made here only sets the shape.
    own_net = copy.deepcopy(shared_net).to(device)
    own_net.eval()
    # The copy above may have been taken while the learner was writing, so it is not trusted:
    # one below any real count makes the first refresh_private_net call a whole copy.
    held_updates = HELD_BEFORE_FIRST_COPY

    # The first `hands` hands of this actor's stream; next_hand_index is the next one to deal.
    slots = []
    next_hand_index = 0
    for _ in range(hands):
        slots.append(deal_slot(seed, actor_id, next_hand_index))
        next_hand_index += 1

    while True:
        held_updates, _ = refresh_private_net(
            own_net, shared_net, version, held_updates, FAST_ACTOR_REFRESH_EVERY
        )
        finished = advance_slots(slots, own_net, epsilon, history_window, device)

        for slot, samples in finished:
            # put waits while the queue is full: the brake that keeps actors from running
            # ahead of the learner, as in actor_loop.
            sample_queue.put((actor_id, slot.hand_index, held_updates, samples))

        # Every finished slot gets the next hand of the stream, so the actor stays at `hands`.
        for position in range(len(slots)):
            if slots[position].state.turn is None:
                slots[position] = deal_slot(seed, actor_id, next_hand_index)
                next_hand_index += 1


def deal_slot(seed, actor_id, hand_index):
    """A HandSlot holding hand `hand_index` of an actor's stream, dealt and ready to play."""
    rng = random.Random(hand_stream_name(seed, actor_id, hand_index))
    return HandSlot(hand_index, rng, new_hand(NUM_PLAYERS, rng))


def train(
    seed,
    num_actors,
    run_dir,
    num_updates,
    samples_per_update=SAMPLES_PER_UPDATE,
    checkpoint_every=CHECKPOINT_EVERY_UPDATES,
    device="cpu",
    shuffle_control=False,
    learning_rate=LEARNING_RATE,
    epsilon=EPSILON,
    optimizer=CONTROL_OPTIMIZER,
    width=DEFAULT_MLP_WIDTH,
    hidden_layers=DEFAULT_HIDDEN_LAYERS,
    history_hidden=DEFAULT_HISTORY_HIDDEN,
    history_window=HISTORY_WINDOW,
    fast_actor=False,
    actor_hands=FAST_ACTOR_HANDS,
    actor_devices=("cpu",),
):
    """
    Runs the training loop with actors in their own processes (spec parts 7-9 and 12).
    seed: int; fixes the starting weights and every actor's hands. The run is still not
      repeatable bit for bit: which actor's hand reaches the queue first depends on timing.
    num_actors: how many actor processes to start.
    run_dir: path of the run's directory, created if missing. It gets log.jsonl, one line per
      update, and checkpoint_<updates done>.pt files, each a QNet state_dict held on the CPU,
      so it loads on a machine with no GPU. checkpoint_000000.pt is the untrained net. A
      directory that already has a log.jsonl is an earlier run and is refused (FileExistsError).
    num_updates: how many learner_step calls to make.
    samples_per_update: batch size B of each update.
    checkpoint_every: a checkpoint is saved after every this many updates, and after the last.
    device: where the learner's net lives and the updates run, e.g. "cpu" or "cuda". The actors
      always play on the CPU.
    shuffle_control: True runs the label-shuffle control: every batch goes through
      shuffle_returns before its update. False (a normal run) leaves the batches alone.
    The rest are what the exploration arms vary (spec parts 16-17); the defaults are the control.
    learning_rate, optimizer: the step size and "rmsprop" or "adam", passed to build_optimizer.
    epsilon: probability that an actor plays a random action on a turn with a choice.
    width, hidden_layers, history_hidden: the net's sizes, passed to QNet for both nets.
    history_window: moves of history the actors encode each decision with (0 = none).
    fast_actor: False runs actor_loop, which plays on one shared CPU net. True runs
      fast_actor_loop (design C, decision log 2026-11-25): each actor scores with its own net
      on one of actor_devices and keeps actor_hands hands in play; the learner writes its new
      weights into one shared net per device after every update (write_weights).
    actor_hands: int, hands each fast actor keeps in play. Unused when fast_actor is False.
    actor_devices: device names the fast actors are spread over, actor a on
      actor_devices[a % len(actor_devices)], e.g. ("cuda:0", "cuda:1"). Unused when fast_actor
      is False.
    Each log line also has actor_lag_mean and actor_lag_max: over the hands this update took,
      how many updates older than the learner's the weights that finished each hand were. Both
      are None when fast_actor is False.
    Returns (q_net, actor_net): the trained learner net, on device, and the shared net the
      actors played with, on the CPU, which holds the same weights.
    Must be called under `if __name__ == "__main__":` in a script: each actor process starts by
      importing the script again.
    """
    # QNet() draws its starting weights from torch's global random stream, so seeding that
    # stream first fixes them. The net is built on the CPU whatever the device: a GPU has its
    # own random stream, and weights drawn there would differ from the CPU's for the same seed.
    torch.manual_seed(seed)
    q_net = QNet(width=width, hidden_layers=hidden_layers, history_hidden=history_hidden)

    # The actors' net: the same sizes and weights as the learner's, moved to memory that every
    # process can read. Actors get a handle to that memory, not a copy of the weights.
    actor_net = QNet(width=width, hidden_layers=hidden_layers, history_hidden=history_hidden)
    actor_net.load_state_dict(q_net.state_dict())
    actor_net.share_memory()

    # "spawn" starts each actor as a fresh Python process. The other way, "fork", copies this
    # process while torch's threads may hold locks, and the copy can then wait on them forever.
    # torch's multiprocessing is the standard library's, plus passing shared tensors by handle.
    context = torch.multiprocessing.get_context("spawn")

    # Fast actors: one net per actor device that the learner writes into and the actors copy
    # from, made while q_net still holds the seeded weights, so every actor starts from them.
    shared_nets = None
    version = None
    if fast_actor:
        shared_nets, version = make_shared_nets(context, list(actor_devices), q_net)

    # Equation 13: the learner's weights move to the device, the actors' copy stays on the CPU.
    # The optimizer is built after the move, so the running averages it keeps per weight are
    # created on the device too.
    q_net.to(device)
    # A local name of its own: the argument `optimizer` is the name, this is the object.
    run_optimizer = build_optimizer(q_net, optimizer, learning_rate)

    # The directory may exist already: the training script makes it first, to write the run's
    # config. A log in it means an earlier run, which is kept, so stop before any actor starts.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        raise FileExistsError(f"{log_path} exists: an earlier run is in this directory")

    # The untrained net, saved before any update: the starting point every later checkpoint is
    # compared with. Checkpoints are saved from the actors' copy because it is on the CPU: a
    # file saved from GPU weights asks for a GPU when it is loaded.
    torch.save(actor_net.state_dict(), run_dir / f"checkpoint_{0:06d}.pt")

    sample_queue = context.Queue(maxsize=QUEUE_MAX_HANDS)
    actors = []
    for actor_id in range(num_actors):
        # daemon: the actor is ended with this process, even if the cleanup below never runs.
        if fast_actor:
            device_index = actor_id % len(actor_devices)
            actor = context.Process(
                target=fast_actor_loop,
                args=(
                    actor_id,
                    seed,
                    actor_devices[device_index],
                    shared_nets[device_index],
                    version,
                    sample_queue,
                    epsilon,
                    history_window,
                    actor_hands,
                ),
                daemon=True,
            )
        else:
            actor = context.Process(
                target=actor_loop,
                args=(actor_id, seed, actor_net, sample_queue, epsilon, history_window),
                daemon=True,
            )
        actor.start()
        actors.append(actor)

    # Its own stream, so the order it draws shares nothing with a hand stream ("train:...") or
    # an evaluation stream. One rng for the whole run: each batch gets the next order from it.
    shuffle_rng = None
    if shuffle_control:
        shuffle_rng = random.Random(f"shuffle:{seed}")

    # Samples received but not yet trained on; what is left after a batch starts the next one.
    waiting_samples = []
    samples_by_actor = []
    for _ in range(num_actors):
        samples_by_actor.append(0)

    try:
        # "x": create the file, and fail if it is there after all.
        with open(log_path, "x") as log_file:
            for update_index in range(num_updates):
                update_start = time.perf_counter()
                # Actor lag of each hand taken this update: the learner has made update_index
                # updates so far, and the hand's last decision was played with weights that
                # include held_updates of them.
                hand_lags = []
                while len(waiting_samples) < samples_per_update:
                    # Raises queue.Empty after the timeout: see QUEUE_TIMEOUT_SECONDS.
                    actor_id, _, held_updates, samples = sample_queue.get(
                        timeout=QUEUE_TIMEOUT_SECONDS
                    )
                    waiting_samples.extend(samples)
                    samples_by_actor[actor_id] += len(samples)
                    if held_updates is not None:
                        hand_lags.append(update_index - held_updates)
                batch = waiting_samples[:samples_per_update]
                waiting_samples = waiting_samples[samples_per_update:]
                if shuffle_control:
                    batch = shuffle_returns(batch, shuffle_rng)

                learner_start = time.perf_counter()
                loss, gradient_norm = learner_step(q_net, run_optimizer, batch)

                # Equation 12: the new weights go where the actors read them. Fast actors copy
                # from one shared net per device; the old actors play on actor_net itself, so
                # its memory is written in place and their next forward pass uses it.
                if fast_actor:
                    write_weights(q_net, shared_nets, version)
                else:
                    actor_net.load_state_dict(q_net.state_dict())
                update_end = time.perf_counter()

                # The loss of a net that has learned nothing: it can do no better than answer
                # the batch's mean return for every sample, and the mean squared error of that
                # answer is the variance of the batch's returns (spec part 13). Dividing by B,
                # not B - 1 (correction=0), is what makes the two equal. Worked out after the
                # timers stop, so it is in neither wait_seconds nor learner_seconds.
                batch_returns = []
                for _, _, _, _, hand_return in batch:
                    batch_returns.append(hand_return)
                return_variance = float(torch.tensor(batch_returns).var(correction=0))

                # None on the old path, where no hand carries a version.
                actor_lag_mean = None
                actor_lag_max = None
                if len(hand_lags) > 0:
                    actor_lag_mean = sum(hand_lags) / len(hand_lags)
                    actor_lag_max = max(hand_lags)

                # wait_seconds is time spent short of samples (the actors are the limit);
                # learner_seconds is the update and the copy (the learner is the limit).
                # Whichever is larger names the bottleneck.
                updates_done = update_index + 1
                log_line = {
                    "update": updates_done,
                    # Samples trained on so far: the x-axis of the learning curve.
                    "samples_total": updates_done * samples_per_update,
                    "loss": loss,
                    "return_variance": return_variance,
                    "gradient_norm": gradient_norm,
                    "wait_seconds": learner_start - update_start,
                    "learner_seconds": update_end - learner_start,
                    "samples_per_second": samples_per_update / (update_end - update_start),
                    "samples_by_actor": list(samples_by_actor),
                    "actor_lag_mean": actor_lag_mean,
                    "actor_lag_max": actor_lag_max,
                }
                log_file.write(json.dumps(log_line) + "\n")
                # flush: the line is on disk now, so a run that dies keeps its log so far.
                log_file.flush()

                if updates_done % checkpoint_every == 0 or updates_done == num_updates:
                    # Saved from actor_net because it is on the CPU. The old path gave it this
                    # update's weights a few lines up; the fast path leaves it alone between
                    # checkpoints (the actors read shared_nets), so it is given them here.
                    if fast_actor:
                        actor_net.load_state_dict(q_net.state_dict())
                    torch.save(
                        actor_net.state_dict(), run_dir / f"checkpoint_{updates_done:06d}.pt"
                    )
    finally:
        # Reached on success, on an error and on Ctrl-C: no actor outlives the run. An actor
        # never returns by itself, so it is ended from here.
        for actor in actors:
            actor.terminate()
        for actor in actors:
            actor.join()
    return q_net, actor_net
