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
- The loop, many processes: actor_loop is what one actor process runs. It plays hands with a
  net it shares with the learner and sends each hand's samples to the learner through a queue.
  train starts the actors, cuts batches from the queue, updates, and after every update copies
  the new weights into the shared net. It writes a log line per update and saves checkpoints.
"""

import json
import random
import time
from pathlib import Path

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


def actor_loop(actor_id, seed, actor_net, sample_queue, epsilon):
    """
    Plays hands forever and queues their samples: the target function of one actor process
    (spec parts 7-8). Never returns; the process that started it ends it.
    actor_id: int, this actor's number, from 0. seed: int, the run's seed.
    actor_net: a QNet whose weights are in shared memory. The learner writes new weights into
      that memory, so this actor's next forward pass uses them without being told.
    sample_queue: a bounded multiprocessing queue. Each item put on it is (actor_id, samples):
      the samples of one finished hand, as self_play_hand gives them.
    epsilon: exploration probability, passed to self_play_hand.
    """
    # By default torch spreads one forward pass over every core. With one actor process per
    # core that would be many times more threads than cores, each waiting on the others.
    torch.set_num_threads(1)

    hands_played = 0
    while True:
        # One rng per hand, named by seed, actor and hand number: no two actors of a run, and
        # no two runs, play the same deal. The "train:" prefix keeps the string different from
        # every evaluation stream.
        hand_rng = random.Random(f"train:{seed}:{actor_id}:{hands_played}")
        state = new_hand(NUM_PLAYERS, hand_rng)
        # The learner may be writing new weights while this hand is played, with no lock, as
        # in DouZero: a decision can be scored by a mix of the old and new weights, which
        # differ by one small update.
        samples, _ = self_play_hand(state, actor_net, epsilon, hand_rng)

        # put waits while the queue is full. That wait is the brake: actors cannot make
        # samples faster than the learner uses them and fill memory.
        # ponytail: the queue copies each hand's samples (about 1 MB) to the learner process.
        # DouZero avoids the copy with shared-memory buffers; switch if the throughput probe
        # shows the queue is the bottleneck.
        sample_queue.put((actor_id, samples))
        hands_played += 1


def train(
    seed,
    num_actors,
    run_dir,
    num_updates,
    samples_per_update=SAMPLES_PER_UPDATE,
    checkpoint_every=CHECKPOINT_EVERY_UPDATES,
):
    """
    Runs the training loop with actors in their own processes (spec parts 7-9).
    seed: int; fixes the starting weights and every actor's hands. The run is still not
      repeatable bit for bit: which actor's hand reaches the queue first depends on timing.
    num_actors: how many actor processes to start.
    run_dir: path of the run's directory, created if missing. It gets log.jsonl, one line per
      update, and checkpoint_<updates done>.pt files, each a QNet state_dict. A directory that
      already has a log.jsonl is an earlier run and is refused (FileExistsError).
    num_updates: how many learner_step calls to make.
    samples_per_update: batch size B of each update.
    checkpoint_every: a checkpoint is saved after every this many updates, and after the last.
    Returns (q_net, actor_net): the trained learner net and the shared net the actors played
      with, which holds the same weights.
    Must be called under `if __name__ == "__main__":` in a script: each actor process starts by
      importing the script again.
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

    # The actors' net: the same weights as the learner's, moved to memory that every process
    # can read. Actors get a handle to that memory, not a copy of the weights.
    actor_net = QNet()
    actor_net.load_state_dict(q_net.state_dict())
    actor_net.share_memory()

    # The directory may exist already: the training script makes it first, to write the run's
    # config. A log in it means an earlier run, which is kept, so stop before any actor starts.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        raise FileExistsError(f"{log_path} exists: an earlier run is in this directory")

    # "spawn" starts each actor as a fresh Python process. The other way, "fork", copies this
    # process while torch's threads may hold locks, and the copy can then wait on them forever.
    # torch's multiprocessing is the standard library's, plus passing shared tensors by handle.
    context = torch.multiprocessing.get_context("spawn")
    sample_queue = context.Queue(maxsize=QUEUE_MAX_HANDS)
    actors = []
    for actor_id in range(num_actors):
        # daemon: the actor is ended with this process, even if the cleanup below never runs.
        actor = context.Process(
            target=actor_loop,
            args=(actor_id, seed, actor_net, sample_queue, EPSILON),
            daemon=True,
        )
        actor.start()
        actors.append(actor)

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
                while len(waiting_samples) < samples_per_update:
                    # Raises queue.Empty after the timeout: see QUEUE_TIMEOUT_SECONDS.
                    actor_id, samples = sample_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
                    waiting_samples.extend(samples)
                    samples_by_actor[actor_id] += len(samples)
                batch = waiting_samples[:samples_per_update]
                waiting_samples = waiting_samples[samples_per_update:]

                learner_start = time.perf_counter()
                loss, gradient_norm = learner_step(q_net, optimizer, batch)

                # Equation 12: write the new weights into the shared memory, in place, so every
                # actor's next forward pass uses them.
                actor_net.load_state_dict(q_net.state_dict())
                update_end = time.perf_counter()

                # wait_seconds is time spent short of samples (the actors are the limit);
                # learner_seconds is the update and the copy (the learner is the limit).
                # Whichever is larger names the bottleneck.
                updates_done = update_index + 1
                log_line = {
                    "update": updates_done,
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "wait_seconds": learner_start - update_start,
                    "learner_seconds": update_end - learner_start,
                    "samples_per_second": samples_per_update / (update_end - update_start),
                    "samples_by_actor": list(samples_by_actor),
                }
                log_file.write(json.dumps(log_line) + "\n")
                # flush: the line is on disk now, so a run that dies keeps its log so far.
                log_file.flush()

                if updates_done % checkpoint_every == 0 or updates_done == num_updates:
                    torch.save(q_net.state_dict(), run_dir / f"checkpoint_{updates_done:06d}.pt")
    finally:
        # Reached on success, on an error and on Ctrl-C: no actor outlives the run. An actor
        # never returns by itself, so it is ended from here.
        for actor in actors:
            actor.terminate()
        for actor in actors:
            actor.join()
    return q_net, actor_net
