"""
Actor-throughput round-trip probe (stage plan specialist_explore, actor-throughput stage piece 1b;
stage 5 step 13b): with real processes and real queues, how fast do actors run when one GPU
server scores their decisions in batches?

The ceiling probe (scripts/throughput_ceiling.py) charged each decision its share of a batched
net call. That holds only if an actor keeps working while it waits for an answer. This probe
measures the real rate instead:
- fake_actor_loop: one actor process. It fakes the game with a busy-wait of the measured engine
  and encoding cost, but sends real decisions over a real queue and waits for real answers. It
  can play several hands at once, so one hand plays while another waits.
- server_loop: one GPU process scoring every queued decision in one batch.
- dummy_learner_loop: real learner updates back to back on the same GPU, to measure how much
  the server and the learner slow each other.
- run_setting: one setting of the grid with real processes (also today's baseline, with no
  server, and the learner alone, with no actors).
- run_speedup and main: each setting's gain for the whole run, and the build verdict.

Two message designs (step 13c, decision log 2026-11-23): per request (the default), and bulk
(--bulk), where an actor sends all of one pass's choice turns as one message and gets one reply
message per server batch. The bulk run adds one per-request row at ANCHOR_HANDS, so both
designs are compared on the same machine on the same day.

Two placements reopened after the bulk no-build (decision log 2026-11-24): several servers
spread over the GPUs (--servers-grid, step 13d), and a net in each actor with no server
(--actor-net, gpu_actor_loop, step 13e).

Run from the repo root (about 10 minutes with the defaults, on teapot):
    PYTHONPATH=src .venv/bin/python scripts/round_trip_probe.py --players 4 --seed 14
    PYTHONPATH=src .venv/bin/python scripts/round_trip_probe.py --players 4 --seed 14 --bulk \
        --hands 1,4,8,16
    PYTHONPATH=src .venv/bin/python scripts/round_trip_probe.py --players 4 --seed 14 --bulk \
        --hands 16 --servers-grid 1:1,2:2,2:1,4:2,8:2
    PYTHONPATH=src .venv/bin/python scripts/round_trip_probe.py --players 4 --seed 14 \
        --actor-net --hands 16,32,64
Output: results/specialist_explore/<config>_round_trip[_bulk]_<machine>_dev_s<seconds>_seed<s>.json

The fake costs come from the ceiling probe's split, so both probes rest on one measurement.

Limits: the engine is a busy-wait, not real play; the learner's weight copy to the server is
left out.
"""

import argparse
import json
import multiprocessing
import os
import queue
import random
import socket
import time
from pathlib import Path

import torch
from evaluate_agents import DEV_SEED_MIN, RESULTS_ROOT, git_state
from throughput_ceiling import (
    BUILD_THRESHOLD,
    DECISION_POOL_SIZE,
    batched_choices,
    choice_decisions,
    parse_sizes,
    untrained_net,
)

from stacker.dmc import (
    EPSILON,
    NUM_PLAYERS,
    SAMPLES_PER_UPDATE,
    build_optimizer,
    hand_stream_name,
    learner_step,
    self_play_hand,
)
from stacker.game import new_hand
from stacker.sealed import config_name

# How often a blocked actor or the server wakes to check whether the run is over.
STOP_CHECK_SECONDS = 0.1
# The most decisions the server scores in one call: the largest batch the ceiling probe timed.
# At 24 actors x 16 hands at most 384 are in flight, so this is only a guard.
MAX_SERVER_BATCH = 1024
# The learner's CPU threads, as in stage 4's teapot runs (train_dmc.py --learner-threads 4).
LEARNER_THREADS = 4
# How long the parent waits for the learner to build its batch before giving up loudly.
LEARNER_READY_TIMEOUT_SECONDS = 120
# How long the parent waits for each process to exit after its result is in; a straggler is
# then terminated (it can only be stuck flushing replies no one reads any more).
JOIN_TIMEOUT_SECONDS = 10
# Slack on top of warm-up and measuring before a missing result counts as a dead process: a
# crashed child never reports, and waiting on it with no limit would hang the probe or a gate.
RESULT_SLACK_SECONDS = 30
MICROSECONDS_PER_SECOND = 1e6
# The round-trip percentiles reported: the typical wait and the slow tail.
MEDIAN = 0.5
P90 = 0.9
# The ceiling probe's teapot result: its split gives the fake actor's costs, and its learner cap
# bounds the run's gain.
DEFAULT_CEILING_JSON = (
    RESULTS_ROOT / "specialist_explore" / "4p1d_throughput_ceiling_teapot_dev_s60_seed14.json"
)
DEFAULT_OUT_DIR = RESULTS_ROOT / "specialist_explore"
DEFAULT_ACTORS = 24
DEFAULT_HANDS = "1,4,16"
DEFAULT_WARMUP_SECONDS = 10.0
DEFAULT_MEASURE_SECONDS = 60.0
# The best result at either of these means the peak may lie between them, so the middle is
# worth adding if it was not run (Nate, 2026-11-23).
HANDS_GAP_LOW = 4
HANDS_GAP_MIDDLE = 8
HANDS_GAP_HIGH = 16
# The bulk run's one per-request row, learner off: the per-request teapot run's 10,493
# decisions/s at 8 hands should come back if the machine is the same as that day.
ANCHOR_HANDS = 8
# The --actor-net run's one learner-off row: the actors' own rate at this many hands, so the
# learner's cost to the actors shows next to the learner-on row (plan, step 13e).
ACTOR_NET_LEARNER_OFF_HANDS = 32
BYTES_PER_GB = 1e9
LIMITS = (
    "engine and encoding faked with a busy-wait of the ceiling split's cost; no weight copy from "
    "learner to server; the round trip includes time an answer waits for its actor to finish "
    "another hand"
)
# Added to LIMITS for an --actor-net run: each actor's net never gets the learner's weights.
ACTOR_NET_LIMITS = "; no weight copy from learner to each actor's net"


def fake_actor_loop(
    actor_id,
    hands,
    decisions,
    kept_seconds,
    choice_share,
    seed,
    request_queue,
    reply_queue,
    measure_event,
    stop_event,
    result_queue,
    record_answers,
    bulk,
):
    """
    One fake actor playing `hands` hands at once until stop_event is set.

    Each hand slot is either ready (it can play its next decision) or waiting (its decision is
    with the server). A ready slot busy-waits kept_seconds, the engine and encoding cost of one
    decision. Then a seeded draw decides the turn's kind:
    - a choice turn (probability choice_share) asks the server for the slot's next pool
      decision and the slot waits;
    - a forced turn needs no net, so it counts as done at once.
    The actor blocks only when every slot is waiting.

    Every message is a list of requests. With bulk off, each choice turn goes out at once as a
    list of one (the per-request probe). With bulk on, the actor plays every ready slot first
    and sends all of that pass's choice turns as one message: one queue put, one pickle, one
    wake-up of the server for many requests.

    actor_id: int, this actor's index; replies come back tagged with it.
    hands: int, hand slots played at once.
    decisions: list of decision tuples as throughput_ceiling.choice_decisions builds them
        (state [S], history [H, HISTORY_DIM], actions [K, ACTION_DIM], ...), sent unchanged.
    kept_seconds: float, busy-wait per decision.
    choice_share: float in [0, 1]; 0 makes the baseline (no server).
    seed: int, seeds the turn-kind draws.
    request_queue: shared queue; each message is (actor_id, [(slot, request_id, decision), ...]).
    reply_queue: this actor's own queue; each message is [(slot, request_id, answer), ...].
    measure_event, stop_event: counting runs only while measure_event is set; the loop ends when
        stop_event is set.
    result_queue: gets one dict at the end: actor_id, decisions (done while measuring),
        round_trips (seconds, one per choice turn), answers ((pool index, answer) pairs if
        record_answers, else empty), spin_seconds (in the busy-wait) and wait_seconds (blocked
        on a reply), both while measuring. The rest of the actor's time is its own queue work.
    record_answers: bool, keep every answer for the cross-talk gate.
    bulk: bool, one message per pass (True) or per choice turn (False).
    """
    rng = random.Random(seed)
    pool_size = len(decisions)

    # Per slot: its next place in the pool, and its open request, None while the slot is ready.
    # Slots start at different places so no two hands send the same decision stream.
    next_index = []
    open_request = []
    for slot in range(hands):
        next_index.append((actor_id * hands + slot) % pool_size)
        open_request.append(None)

    request_id = 0
    decisions_done = 0
    round_trips = []
    answers = []
    spin_seconds = 0.0
    wait_seconds = 0.0

    while not stop_event.is_set():
        # Every ready slot plays one decision. A choice turn's request goes out now (bulk off)
        # or joins this pass's message (bulk on).
        outgoing = []
        for slot in range(hands):
            if open_request[slot] is not None:
                continue
            # Busy-wait, not sleep: the real engine keeps its core busy for this long.
            spin_start = time.perf_counter()
            spin_until = spin_start + kept_seconds
            while time.perf_counter() < spin_until:
                pass
            if measure_event.is_set():
                spin_seconds += time.perf_counter() - spin_start
            if rng.random() < choice_share:
                pool_index = next_index[slot]
                next_index[slot] = (pool_index + 1) % pool_size
                request = (slot, request_id, decisions[pool_index])
                if bulk:
                    outgoing.append(request)
                else:
                    request_queue.put((actor_id, [request]))
                # The clock starts when the request is made, so in bulk mode the round trip
                # includes its wait for the rest of the pass.
                open_request[slot] = (request_id, pool_index, time.perf_counter())
                request_id += 1
            else:
                if measure_event.is_set():
                    decisions_done += 1
        if len(outgoing) > 0:
            request_queue.put((actor_id, outgoing))

        # Block for a reply only when no slot can play; the timeout lets the loop see stop_event.
        all_waiting = True
        for slot in range(hands):
            if open_request[slot] is None:
                all_waiting = False
        replies = []
        if all_waiting:
            wait_start = time.perf_counter()
            try:
                replies.append(reply_queue.get(timeout=STOP_CHECK_SECONDS))
            except queue.Empty:
                pass
            if measure_event.is_set():
                wait_seconds += time.perf_counter() - wait_start
        # Then take every reply already queued, without blocking.
        while True:
            try:
                replies.append(reply_queue.get_nowait())
            except queue.Empty:
                break

        # Each reply frees its slot. Its id must match the slot's open request: a reply meant
        # for another request is cross-talk, and the probe stops rather than count it.
        for message in replies:
            for slot, replied_id, answer in message:
                sent_id, pool_index, sent_at = open_request[slot]
                if replied_id != sent_id:
                    raise RuntimeError(
                        f"actor {actor_id} slot {slot}: reply {replied_id}, open request {sent_id}"
                    )
                open_request[slot] = None
                if measure_event.is_set():
                    decisions_done += 1
                    round_trips.append(time.perf_counter() - sent_at)
                    if record_answers:
                        answers.append((pool_index, answer))

    result_queue.put(
        {
            "actor_id": actor_id,
            "decisions": decisions_done,
            "round_trips": round_trips,
            "answers": answers,
            "spin_seconds": spin_seconds,
            "wait_seconds": wait_seconds,
        }
    )


def gpu_actor_loop(
    actor_id,
    hands,
    decisions,
    kept_seconds,
    choice_share,
    seed,
    device_name,
    measure_event,
    stop_event,
    result_queue,
    record_answers,
):
    """
    One fake actor that scores its own decisions on its own GPU, playing `hands` hands at once
    until stop_event is set: the net-in-each-actor design (step 13e), with no server and no
    queues.

    Each pass, every hand plays one decision: a busy-wait of kept_seconds, then the same seeded
    draw as fake_actor_loop decides the turn's kind. A forced turn counts as done at once. A
    choice turn's decision joins this pass's list. At the end of the pass, one batched_choices
    call scores the whole list on this actor's device, and every hand is ready again. So the
    batch is this actor's choice turns of one pass, about choice_share * hands decisions.

    DouZero's actors also hold the model on a GPU, but call it once per decision; batching
    across hands is this probe's addition (decision log 2026-11-24).

    actor_id, hands, decisions, kept_seconds, choice_share, seed, measure_event, stop_event,
        record_answers: as in fake_actor_loop.
    device_name: str, this actor's device, e.g. "cuda:1"; the caller spreads actors over GPUs.
    result_queue: gets one dict at the end with fake_actor_loop's keys, so run_setting reads it
        unchanged: round_trips are from a choice turn's draw to its answer (the rest of the
        pass plus the net call), wait_seconds is 0 (the actor never blocks). Two more keys,
        both while measuring: net_seconds (in batched_choices) and net_calls.
    """
    # One CPU thread per actor: 24 actors must not fight over the cores for torch's threads.
    torch.set_num_threads(1)
    device = torch.device(device_name)
    net = untrained_net().to(device)
    net.eval()

    rng = random.Random(seed)
    pool_size = len(decisions)
    # Each hand's next place in the pool; hands start at different places, as in
    # fake_actor_loop.
    next_index = []
    for slot in range(hands):
        next_index.append((actor_id * hands + slot) % pool_size)

    decisions_done = 0
    round_trips = []
    answers = []
    spin_seconds = 0.0
    net_seconds = 0.0
    net_calls = 0

    while not stop_event.is_set():
        # Every hand plays one decision; choice turns wait for the pass's one net call.
        batch_indices = []
        batch_started = []
        for slot in range(hands):
            # Busy-wait, not sleep: the real engine keeps its core busy for this long.
            spin_start = time.perf_counter()
            spin_until = spin_start + kept_seconds
            while time.perf_counter() < spin_until:
                pass
            if measure_event.is_set():
                spin_seconds += time.perf_counter() - spin_start
            if rng.random() < choice_share:
                pool_index = next_index[slot]
                next_index[slot] = (pool_index + 1) % pool_size
                batch_indices.append(pool_index)
                batch_started.append(time.perf_counter())
            else:
                if measure_event.is_set():
                    decisions_done += 1

        if len(batch_indices) == 0:
            continue
        # One call for the pass: answer i belongs to batch_indices[i], same order in and out.
        batch = []
        for pool_index in batch_indices:
            batch.append(decisions[pool_index])
        net_start = time.perf_counter()
        batch_answers = batched_choices(net, batch, device)
        net_end = time.perf_counter()

        if measure_event.is_set():
            net_seconds += net_end - net_start
            net_calls += 1
            decisions_done += len(batch_indices)
            for row in range(len(batch_indices)):
                round_trips.append(net_end - batch_started[row])
                if record_answers:
                    answers.append((batch_indices[row], batch_answers[row]))

    result_queue.put(
        {
            "actor_id": actor_id,
            "decisions": decisions_done,
            "round_trips": round_trips,
            "answers": answers,
            "spin_seconds": spin_seconds,
            "wait_seconds": 0.0,
            "net_seconds": net_seconds,
            "net_calls": net_calls,
        }
    )


def server_loop(
    request_queue, reply_queues, device_name, measure_event, stop_event, result_queue, bulk
):
    """
    The GPU server: scores every queued decision with one batched net call and sends each answer
    to the actor that asked, until stop_event is set.

    It waits for one message, then takes every other message already queued, with no extra wait,
    until it holds MAX_SERVER_BATCH requests (one large message can carry it a little past).
    Each message's list is unpacked into rows. batched_choices stacks the rows, runs one forward
    pass, masks padded actions and takes each decision's argmax. Row i of the batch is request i,
    so answer i goes back with request i's actor, slot and id: one message per request with
    bulk off, one message per actor with bulk on.

    Its time is split three ways, to find which step limits it:
    - drain: taking and unpacking the messages already queued. The wait for the first message
      is left out, because it includes idle time; so the per-request receive cost is drain time
      over the requests that arrived after the first message.
    - net: batched_choices.
    - reply: building and sending the replies.

    request_queue: shared queue; each message is (actor_id, [(slot, request_id, decision), ...]).
    reply_queues: list of queues, one per actor, indexed by actor_id; each message is
        [(slot, request_id, answer), ...], answer an int row of the decision's action list.
    device_name: str, "cuda" or "cpu".
    measure_event, stop_event: as in fake_actor_loop.
    result_queue: gets one dict at the end, all counted while measuring: batch_sizes (one per
        call), messages_received, requests_received, drained_requests, drain_seconds,
        net_seconds, reply_seconds.
    bulk: bool, reply with one message per actor (True) or per request (False).
    """
    # One CPU thread: stacking the batch must not take cores from the actors.
    torch.set_num_threads(1)
    device = torch.device(device_name)
    net = untrained_net().to(device)
    net.eval()

    batch_sizes = []
    messages_received = 0
    requests_received = 0
    drained_requests = 0
    drain_seconds = 0.0
    net_seconds = 0.0
    reply_seconds = 0.0

    while not stop_event.is_set():
        # Wait for the first message; the timeout lets the loop see stop_event.
        try:
            messages = [request_queue.get(timeout=STOP_CHECK_SECONDS)]
        except queue.Empty:
            continue
        # Then take every message already queued, without waiting for more.
        drain_start = time.perf_counter()
        first_message_requests = len(messages[0][1])
        request_count = first_message_requests
        while request_count < MAX_SERVER_BATCH:
            try:
                message = request_queue.get_nowait()
            except queue.Empty:
                break
            messages.append(message)
            request_count += len(message[1])

        # One row per request, remembering which actor, slot and id it came from.
        row_actors = []
        row_slots = []
        row_ids = []
        decisions = []
        for actor_id, requests in messages:
            for slot, request_id, decision in requests:
                row_actors.append(actor_id)
                row_slots.append(slot)
                row_ids.append(request_id)
                decisions.append(decision)

        net_start = time.perf_counter()
        answers = batched_choices(net, decisions, device)
        reply_start = time.perf_counter()

        # Answer i belongs to request i: same order in, same order out.
        if bulk:
            # Group each actor's answers, so each actor gets one message per batch.
            replies_by_actor = {}
            for row in range(len(decisions)):
                actor_id = row_actors[row]
                if actor_id not in replies_by_actor:
                    replies_by_actor[actor_id] = []
                replies_by_actor[actor_id].append((row_slots[row], row_ids[row], answers[row]))
            for actor_id in replies_by_actor:
                reply_queues[actor_id].put(replies_by_actor[actor_id])
        else:
            for row in range(len(decisions)):
                reply_queues[row_actors[row]].put([(row_slots[row], row_ids[row], answers[row])])
        reply_end = time.perf_counter()

        if measure_event.is_set():
            batch_sizes.append(len(decisions))
            messages_received += len(messages)
            requests_received += len(decisions)
            drained_requests += len(decisions) - first_message_requests
            drain_seconds += net_start - drain_start
            net_seconds += reply_start - net_start
            reply_seconds += reply_end - reply_start

    result_queue.put(
        {
            "batch_sizes": batch_sizes,
            "messages_received": messages_received,
            "requests_received": requests_received,
            "drained_requests": drained_requests,
            "drain_seconds": drain_seconds,
            "net_seconds": net_seconds,
            "reply_seconds": reply_seconds,
        }
    )


def dummy_learner_loop(seed, device_name, ready_event, measure_event, stop_event, result_queue):
    """
    A learner that does real updates back to back until stop_event is set, on one fixed batch.

    It plays real hands with the untrained net until it has SAMPLES_PER_UPDATE samples, then
    calls learner_step on that batch over and over: collate on the CPU, move to the device,
    forward, backward, clip, RMSprop step. The weights drift, but an update costs the same
    whatever their values. Back to back is the busiest the learner gets, which a faster actor
    side would make it.

    seed: int, dev seed naming the hand streams.
    device_name: str, "cuda" or "cpu".
    ready_event: set once the batch is built, so the caller can start the actors after setup.
    measure_event, stop_event: as in fake_actor_loop.
    result_queue: gets one dict at the end: updates and update_seconds, both while measuring.
    """
    torch.set_num_threads(LEARNER_THREADS)
    net = untrained_net()

    # One real batch, played on the CPU as train_one_process plays it: real hands give the real
    # mix of action counts, which sets the cost of collating and padding.
    samples = []
    hands_played = 0
    while len(samples) < SAMPLES_PER_UPDATE:
        hand_rng = random.Random(hand_stream_name(seed, 0, hands_played))
        state = new_hand(NUM_PLAYERS, hand_rng)
        hand_samples, _ = self_play_hand(state, net, EPSILON, hand_rng)
        samples.extend(hand_samples)
        hands_played += 1
    batch = samples[:SAMPLES_PER_UPDATE]

    net = net.to(torch.device(device_name))
    optimizer = build_optimizer(net)
    ready_event.set()

    updates = 0
    update_seconds = 0.0
    while not stop_event.is_set():
        update_start = time.perf_counter()
        # learner_step returns the loss as a Python float, which waits for the device to
        # finish: the time below covers the whole update.
        learner_step(net, optimizer, batch)
        if measure_event.is_set():
            updates += 1
            update_seconds += time.perf_counter() - update_start

    result_queue.put({"updates": updates, "update_seconds": update_seconds})


def percentile(sorted_values, fraction):
    """The value at `fraction` of the way through an ascending list (nearest rank below)."""
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def combine_server_results(server_results):
    """
    Merges the servers' end-of-run dicts into one of the same shape, as if one server had
    done all their work: batch_sizes joined, every count and every seconds total summed.
    server_results: list of server_loop result dicts, one per server.
    Returns that dict plus servers_used, the number of servers that scored at least one
    request while measuring (fewer than started means some servers got no actors' requests).
    """
    summed_fields = [
        "messages_received",
        "requests_received",
        "drained_requests",
        "drain_seconds",
        "net_seconds",
        "reply_seconds",
    ]
    combined = {"batch_sizes": [], "servers_used": 0}
    for field in summed_fields:
        combined[field] = 0
    for result in server_results:
        combined["batch_sizes"].extend(result["batch_sizes"])
        for field in summed_fields:
            combined[field] += result[field]
        if result["requests_received"] > 0:
            combined["servers_used"] += 1
    return combined


def run_setting(
    num_actors,
    hands,
    learner_on,
    decisions,
    kept_seconds,
    choice_share,
    device_name,
    seed,
    warmup_seconds,
    measure_seconds,
    record_answers,
    bulk,
    num_servers=1,
    num_gpus=1,
    actor_net=False,
):
    """
    Runs one setting with real processes and returns its rates.

    Processes: num_actors fake actors; num_servers servers if there are actors with choice
    turns (choice_share > 0); one dummy learner if learner_on. So the same function gives
    - today's baseline: choice_share 0, kept_seconds = today's whole cost per decision;
    - a grid setting: choice_share and kept_seconds from the split, with a server;
    - the learner alone: num_actors 0, learner_on.
    The learner starts first and builds its batch before anything else starts. After
    warmup_seconds the parent sets measure_event, and measure_seconds later stop_event.

    decisions: the pool, as in fake_actor_loop. seed: dev seed; actor i seeds its draws with
    seed + i, the learner names its hand streams with seed. bulk: as in fake_actor_loop.
    num_servers: N; actor a sends to server a % N, through that server's own request queue,
    so no two servers take each other's requests. Each server holds every actor's reply
    queue and indexes it by the global actor id carried in each message, so its replies
    reach the actor that asked.
    num_gpus: G; server s runs on device_name:{s % G}. The learner stays on device_name, which
    for "cuda" is GPU 0. G must be 1 on the CPU.
    actor_net: True runs gpu_actor_loop actors instead (step 13e): actor a scores its own
    decisions on device_name:{a % G}, no server starts, and num_servers and bulk are unused.
    Returns a dict: the inputs, measured_seconds, decisions_per_second, the median and p90
    round trip in microseconds (it includes the time an answer waits for its actor to finish
    another hand), servers_used, mean_batch over every server's calls, server_busy_share (net
    plus reply, as in the per-request run), the servers' split (drain and reply microseconds
    per request, net share, requests per message; shares are of all servers' time together,
    so 1.0 means every server busy all the time), the actors' split (spin, wait and other shares of their time),
    learner_updates_per_second, and answers (one list per actor, filled only if
    record_answers). With actor_net, servers_used is 0, mean_batch is over the actors' own net
    calls, actor_net_share is the actors' time in those calls (part of actor_other_share), and
    gpu_used_gb lists each GPU's memory in use near the end of the window (cuda only).
    """
    # spawn, as train uses: a forked child would inherit torch's threads mid-state.
    context = multiprocessing.get_context("spawn")
    request_queues = []
    for server_id in range(num_servers):
        request_queues.append(context.Queue())
    result_queue = context.Queue()
    measure_event = context.Event()
    stop_event = context.Event()
    ready_event = context.Event()
    reply_queues = []
    for actor_id in range(num_actors):
        reply_queues.append(context.Queue())

    processes = []
    if learner_on:
        learner = context.Process(
            target=dummy_learner_loop,
            args=(seed, device_name, ready_event, measure_event, stop_event, result_queue),
        )
        learner.start()
        processes.append(learner)
        if not ready_event.wait(LEARNER_READY_TIMEOUT_SECONDS):
            raise RuntimeError("the dummy learner did not build its batch in time")

    server_on = False
    if num_actors > 0 and choice_share > 0 and not actor_net:
        server_on = True
        for server_id in range(num_servers):
            # Spread the servers over the GPUs in turn: with 4 servers on 2 GPUs, servers 0
            # and 2 share GPU 0. One GPU keeps the bare name, as the one-server run did.
            server_device = device_name
            if num_gpus > 1:
                server_device = f"{device_name}:{server_id % num_gpus}"
            server = context.Process(
                target=server_loop,
                args=(
                    request_queues[server_id], reply_queues, server_device, measure_event,
                    stop_event, result_queue, bulk,
                ),
            )
            server.start()
            processes.append(server)

    for actor_id in range(num_actors):
        if actor_net:
            # Spread the actors over the GPUs in turn, as the servers are spread above.
            actor_device = device_name
            if num_gpus > 1:
                actor_device = f"{device_name}:{actor_id % num_gpus}"
            actor = context.Process(
                target=gpu_actor_loop,
                args=(
                    actor_id, hands, decisions, kept_seconds, choice_share, seed + actor_id,
                    actor_device, measure_event, stop_event, result_queue, record_answers,
                ),
            )
            actor.start()
            processes.append(actor)
            continue
        actor = context.Process(
            target=fake_actor_loop,
            args=(
                actor_id,
                hands,
                decisions,
                kept_seconds,
                choice_share,
                seed + actor_id,
                request_queues[actor_id % num_servers],
                reply_queues[actor_id],
                measure_event,
                stop_event,
                result_queue,
                record_answers,
                bulk,
            ),
        )
        actor.start()
        processes.append(actor)

    # Warm-up covers process start-up and the first device calls; only the window after it
    # counts. Its length is timed here, in one process, so no clocks are compared across
    # processes.
    time.sleep(warmup_seconds)
    measure_event.set()
    measure_start = time.perf_counter()
    time.sleep(measure_seconds)
    # Net in each actor: read each GPU's memory in use while every actor still holds its net.
    # It counts everything on the GPU, the learner and other users' processes included.
    gpu_used_bytes = []
    if actor_net and device_name.startswith("cuda"):
        for gpu in range(num_gpus):
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
            gpu_used_bytes.append(total_bytes - free_bytes)
    stop_event.set()
    measured_seconds = time.perf_counter() - measure_start

    # Results are read before joining: a child that has put data on a queue cannot exit until
    # that data is read, so joining first would wait forever.
    actor_results = []
    server_results = []
    learner_result = None
    result_timeout = warmup_seconds + measure_seconds + RESULT_SLACK_SECONDS
    for _ in range(len(processes)):
        try:
            result = result_queue.get(timeout=result_timeout)
        except queue.Empty:
            for process in processes:
                process.terminate()
            raise RuntimeError("a probe process died before reporting")
        if "actor_id" in result:
            actor_results.append(result)
        elif "batch_sizes" in result:
            server_results.append(result)
        else:
            learner_result = result
    for process in processes:
        process.join(JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()

    total_decisions = 0
    round_trips = []
    answers = []
    total_spin_seconds = 0.0
    total_wait_seconds = 0.0
    # Only gpu_actor_loop actors report their own net time and calls.
    total_actor_net_seconds = 0.0
    total_actor_net_calls = 0
    for result in sorted(actor_results, key=lambda item: item["actor_id"]):
        if actor_net:
            total_actor_net_seconds += result["net_seconds"]
            total_actor_net_calls += result["net_calls"]
        total_decisions += result["decisions"]
        round_trips.extend(result["round_trips"])
        answers.append(result["answers"])
        total_spin_seconds += result["spin_seconds"]
        total_wait_seconds += result["wait_seconds"]
    round_trips.sort()
    # The servers' work is summed; their time shares are over all servers' time together.
    server_result = None
    server_seconds = len(server_results) * measured_seconds
    if server_results:
        server_result = combine_server_results(server_results)

    summary = {
        "num_actors": num_actors,
        "hands": hands,
        "learner_on": learner_on,
        "server_on": server_on,
        "kept_seconds": kept_seconds,
        "choice_share": choice_share,
        "device": device_name,
        "seed": seed,
        "warmup_seconds": warmup_seconds,
        "bulk": bulk,
        "num_servers": num_servers,
        "num_gpus": num_gpus,
        "actor_net": actor_net,
        "servers_used": None,
        "measured_seconds": measured_seconds,
        "decisions_per_second": total_decisions / measured_seconds,
        "round_trip_incl_actor_wait_median_us": None,
        "round_trip_incl_actor_wait_p90_us": None,
        "mean_batch": None,
        "server_busy_share": None,
        "server_net_share": None,
        "server_drain_us_per_request": None,
        "server_reply_us_per_request": None,
        "requests_per_message": None,
        "actor_spin_share": None,
        "actor_wait_share": None,
        "actor_other_share": None,
        "actor_net_share": None,
        "gpu_used_gb": [],
        "learner_updates_per_second": None,
        "answers": answers,
    }
    if round_trips:
        summary["round_trip_incl_actor_wait_median_us"] = (
            percentile(round_trips, MEDIAN) * MICROSECONDS_PER_SECOND
        )
        summary["round_trip_incl_actor_wait_p90_us"] = (
            percentile(round_trips, P90) * MICROSECONDS_PER_SECOND
        )
    if server_result is not None and server_result["batch_sizes"]:
        batch_sizes = server_result["batch_sizes"]
        summary["servers_used"] = server_result["servers_used"]
        summary["mean_batch"] = sum(batch_sizes) / len(batch_sizes)
        summary["server_busy_share"] = (
            server_result["net_seconds"] + server_result["reply_seconds"]
        ) / server_seconds
        summary["server_net_share"] = server_result["net_seconds"] / server_seconds
        summary["server_reply_us_per_request"] = (
            server_result["reply_seconds"] / server_result["requests_received"]
        ) * MICROSECONDS_PER_SECOND
        summary["requests_per_message"] = (
            server_result["requests_received"] / server_result["messages_received"]
        )
        # Every call's first message is left out of the drain; with one message per call
        # nothing was drained.
        if server_result["drained_requests"] > 0:
            summary["server_drain_us_per_request"] = (
                server_result["drain_seconds"] / server_result["drained_requests"]
            ) * MICROSECONDS_PER_SECOND
    # The actors' time: spinning (fake game work), blocked on a reply, and the rest, which is
    # their own queue work (pickling, putting, unpickling) and loop overhead.
    if num_actors > 0:
        actor_seconds = num_actors * measured_seconds
        summary["actor_spin_share"] = total_spin_seconds / actor_seconds
        summary["actor_wait_share"] = total_wait_seconds / actor_seconds
        summary["actor_other_share"] = 1.0 - (total_spin_seconds + total_wait_seconds) / actor_seconds
    # Net in each actor: one round trip is kept per choice turn, so choice turns over net calls
    # is the mean batch.
    if actor_net:
        summary["servers_used"] = 0
        summary["actor_net_share"] = total_actor_net_seconds / (num_actors * measured_seconds)
        for used_bytes in gpu_used_bytes:
            summary["gpu_used_gb"].append(used_bytes / BYTES_PER_GB)
        if total_actor_net_calls > 0:
            summary["mean_batch"] = len(round_trips) / total_actor_net_calls
    if learner_result is not None:
        summary["learner_updates_per_second"] = learner_result["updates"] / measured_seconds
    return summary


def run_speedup(actor_speedup, learner_cap, learner_slowdown):
    """
    The whole run's gain: the actors' gain, but no more than the learner allows.
    learner_cap: the run's most gain with infinitely fast actors (the ceiling probe's 3.56).
    learner_slowdown: the learner's updates/s alone over its updates/s next to the server (1 if
    the learner is off). A learner slowed by the server caps the run lower by the same factor.
    """
    cap = learner_cap / learner_slowdown
    if actor_speedup < cap:
        return actor_speedup
    else:
        return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--ceiling-json", type=Path, default=DEFAULT_CEILING_JSON)
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--actors", type=int, default=DEFAULT_ACTORS)
    parser.add_argument("--hands", default=DEFAULT_HANDS, help="hands per actor, comma list")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--seconds", type=float, default=DEFAULT_MEASURE_SECONDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--bulk", action="store_true", help="one message per actor pass, not per choice turn"
    )
    parser.add_argument(
        "--servers-grid",
        default=None,
        help="servers:GPUs pairs, comma list (e.g. 1:1,2:2,2:1,4:2,8:2); learner-on rows only",
    )
    parser.add_argument(
        "--actor-net",
        action="store_true",
        help="each actor scores its own decisions on its own GPU share; no server, learner-on rows",
    )
    args = parser.parse_args()
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )
    # The net-in-each-actor design has no server, so no message design and no servers grid.
    if args.actor_net and (args.bulk or args.servers_grid is not None):
        raise SystemExit("--actor-net runs without --bulk or --servers-grid")

    # The fake actor's costs, per decision, from the ceiling probe's measured split.
    with open(args.ceiling_json) as file:
        ceiling = json.load(file)
    split = ceiling["split"]
    kept_seconds = (split["engine_and_other_seconds"] + split["encode_seconds"]) / split["decisions"]
    today_seconds = split["wall_seconds"] / split["decisions"]
    choice_share = split["choice_turns"] / split["decisions"]
    learner_cap = ceiling["learner_cap"]
    print(
        f"per decision: kept {kept_seconds * MICROSECONDS_PER_SECOND:.0f} us, today "
        f"{today_seconds * MICROSECONDS_PER_SECOND:.0f} us; choice share {choice_share:.2f}; "
        f"learner cap {learner_cap:.2f}"
    )

    # Every process wants a core: actors, the server and the learner's threads. Count the cores
    # this process may run on (39 of teapot's 48); macOS has no affinity call, so count them all.
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count()
    cores_short = cores < args.actors + 1 + LEARNER_THREADS
    print(f"cores: {cores}")
    if cores_short:
        print("warning: fewer cores than processes; rates will read low")

    # The servers grid: each "servers:GPUs" pair becomes (servers, GPUs). Without it, every
    # row has one server on one GPU, as the earlier round-trip runs had.
    server_settings = [(1, 1)]
    if args.servers_grid is not None:
        server_settings = []
        for pair in args.servers_grid.split(","):
            servers_text, gpus_text = pair.split(":")
            server_settings.append((int(servers_text), int(gpus_text)))
    # Check the GPU counts before any run starts, not halfway through the grid.
    gpus_available = 1
    if args.device.startswith("cuda"):
        gpus_available = torch.cuda.device_count()
    for num_servers, num_gpus in server_settings:
        if num_gpus > gpus_available:
            raise SystemExit(f"{num_gpus} GPUs asked for, {gpus_available} available")

    decisions = choice_decisions(args.players, args.seed, DECISION_POOL_SIZE)

    def run(
        num_actors, hands, learner_on, kept, share, bulk, num_servers=1, num_gpus=1,
        actor_net=False,
    ):
        return run_setting(
            num_actors, hands, learner_on, decisions, kept, share, args.device, args.seed,
            args.warmup, args.seconds, False, bulk, num_servers, num_gpus, actor_net,
        )

    def add_speedups(row, learner_on):
        # The row's gains: actors against today's baseline, and the whole run's, capped by the
        # learner as slowed by the server.
        row.pop("answers")
        row["actor_speedup"] = row["decisions_per_second"] / baseline_rate
        learner_slowdown = 1.0
        if learner_on:
            learner_slowdown = alone_updates / row["learner_updates_per_second"]
        row["learner_slowdown"] = learner_slowdown
        row["run_speedup"] = run_speedup(row["actor_speedup"], learner_cap, learner_slowdown)

    def print_row(label, row):
        drain = "n/a"
        if row["server_drain_us_per_request"] is not None:
            drain = f"{row['server_drain_us_per_request']:.0f} us"
        print(
            f"{label}: {row['decisions_per_second']:,.0f} decisions/s, "
            f"actor x{row['actor_speedup']:.2f}, run x{row['run_speedup']:.2f}, "
            f"mean batch {row['mean_batch']:.1f}, "
            f"round trip median {row['round_trip_incl_actor_wait_median_us']:.0f} us, "
            f"learner slowdown x{row['learner_slowdown']:.2f}"
        )
        # With the net in each actor there is no server line; the actors' net share replaces it.
        if row["actor_net"]:
            memory = []
            for used_gb in row["gpu_used_gb"]:
                memory.append(f"{used_gb:.1f}")
            print(
                f"  actors: spin {row['actor_spin_share']:.0%}, net "
                f"{row['actor_net_share']:.0%}, other {row['actor_other_share']:.0%}; "
                f"GPU memory in use (GB): {', '.join(memory)}"
            )
            return
        print(
            f"  server: net {row['server_net_share']:.0%}, drain {drain}/request, "
            f"reply {row['server_reply_us_per_request']:.0f} us/request, "
            f"{row['requests_per_message']:.1f} requests/message; actors: spin "
            f"{row['actor_spin_share']:.0%}, wait {row['actor_wait_share']:.0%}, "
            f"other {row['actor_other_share']:.0%}"
        )

    learner_alone = run(0, 1, True, kept_seconds, choice_share, False)
    alone_updates = learner_alone["learner_updates_per_second"]
    print(f"learner alone: {alone_updates:.2f} updates/s")
    baseline = run(args.actors, 1, False, today_seconds, 0.0, False)
    baseline_rate = baseline["decisions_per_second"]
    print(f"today's baseline: {baseline_rate:,.0f} decisions/s")

    # The bulk run's per-request anchor row, learner off. It is not in the verdict. The servers
    # grid leaves it out: its own one-server row is the anchor there.
    anchor = None
    if args.bulk and args.servers_grid is None:
        anchor = run(args.actors, ANCHOR_HANDS, False, kept_seconds, choice_share, False)
        add_speedups(anchor, False)
        print_row(f"anchor, per request, hands {ANCHOR_HANDS}, learner off", anchor)

    hands_list = parse_sizes(args.hands)
    rows = []
    if args.actor_net:
        # Every actor holds its own net, spread over every GPU. The learner-on rows are what the
        # verdict reads nothing else. One learner-off row shows the learner's cost to the actors.
        actor_net_settings = []
        for hands in hands_list:
            actor_net_settings.append((hands, True))
        actor_net_settings.append((ACTOR_NET_LEARNER_OFF_HANDS, False))
        for hands, learner_on in actor_net_settings:
            row = run(
                args.actors, hands, learner_on, kept_seconds, choice_share, False, 0,
                gpus_available, True,
            )
            add_speedups(row, learner_on)
            rows.append(row)
            learner_label = "off"
            if learner_on:
                learner_label = "on"
            print_row(
                f"hands {hands}, net in each actor on {gpus_available} GPUs, "
                f"learner {learner_label}",
                row,
            )
    elif args.servers_grid is None:
        for hands in hands_list:
            for learner_on in [False, True]:
                row = run(args.actors, hands, learner_on, kept_seconds, choice_share, args.bulk)
                add_speedups(row, learner_on)
                rows.append(row)
                learner_label = "off"
                if learner_on:
                    learner_label = "on"
                print_row(f"hands {hands}, learner {learner_label}", row)
    else:
        # The servers grid runs learner on only: the verdict reads nothing else.
        for hands in hands_list:
            for num_servers, num_gpus in server_settings:
                row = run(
                    args.actors, hands, True, kept_seconds, choice_share, args.bulk,
                    num_servers, num_gpus,
                )
                add_speedups(row, True)
                rows.append(row)
                print_row(
                    f"hands {hands}, servers {num_servers} on {num_gpus} GPUs "
                    f"({row['servers_used']} used), learner on",
                    row,
                )

    # The verdict reads only the learner-on rows: in a real run the learner is always there.
    best_row = None
    for row in rows:
        if row["learner_on"]:
            if best_row is None or row["run_speedup"] > best_row["run_speedup"]:
                best_row = row
    if best_row["run_speedup"] < BUILD_THRESHOLD:
        verdict = "no build"
    else:
        verdict = "build"
    placement = f"servers {best_row['num_servers']}"
    if best_row["actor_net"]:
        placement = "net in each actor"
    print(
        f"best learner-on: hands {best_row['hands']}, {placement} on "
        f"{best_row['num_gpus']} GPUs, run x{best_row['run_speedup']:.2f}; verdict: {verdict}"
    )
    # The best at 4 or at 16 means the peak may sit between them, if nothing was run there.
    best_at_gap_edge = best_row["hands"] == HANDS_GAP_LOW or best_row["hands"] == HANDS_GAP_HIGH
    if best_at_gap_edge and HANDS_GAP_MIDDLE not in hands_list:
        print(
            f"note: the best is at {best_row['hands']} and {HANDS_GAP_MIDDLE} is untested; "
            f"consider adding --hands {HANDS_GAP_MIDDLE}"
        )

    learner_alone.pop("answers")
    baseline.pop("answers")
    commit, uncommitted_changes = git_state()
    machine = socket.gethostname().split(".")[0]
    design = ""
    if args.bulk:
        design = "_bulk"
    if args.servers_grid is not None:
        design += "_servers"
    limits = LIMITS
    if args.actor_net:
        design += "_actor_net"
        limits += ACTOR_NET_LIMITS
    run_name = (
        f"{config_name(args.players)}_round_trip{design}_{machine}_dev_s{args.seconds:g}"
        f"_seed{args.seed}"
    )
    record = {
        "players": args.players,
        "seed": args.seed,
        "source": "dev",
        "machine": machine,
        "device": args.device,
        "cores": cores,
        "cores_short": cores_short,
        "ceiling_json": str(args.ceiling_json),
        "kept_seconds": kept_seconds,
        "today_seconds": today_seconds,
        "choice_share": choice_share,
        "learner_cap": learner_cap,
        "learner_alone": learner_alone,
        "baseline": baseline,
        "bulk": args.bulk,
        "anchor": anchor,
        "server_settings": server_settings,
        "actor_net": args.actor_net,
        "rows": rows,
        "best_hands": best_row["hands"],
        "best_num_servers": best_row["num_servers"],
        "best_num_gpus": best_row["num_gpus"],
        "best_run_speedup": best_row["run_speedup"],
        "build_threshold": BUILD_THRESHOLD,
        "verdict": verdict,
        "limits": limits,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (run_name + ".json")
    with open(out_path, "w") as file:
        json.dump(record, file, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
