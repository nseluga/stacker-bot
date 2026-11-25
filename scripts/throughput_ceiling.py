"""
Actor-throughput ceiling probe (stage plan specialist_explore, actor-throughput stage piece 1;
stage 5 piece 13): could batching the actors' net calls make training at least 1.5x faster?

An actor today scores one decision per net call. The probe measures three things and combines
them:
- actor_time_split: where one actor's time goes (engine, encoding, the net call), playing real
  hands with the untrained control net.
- net_decisions_per_second: what one decision costs when B decisions share one net call, on one
  CPU core and on the GPU. The cost covers stacking the batch, moving it, the forward pass and
  picking each decision's best real action.
- learner_cap: training cannot go faster than the learner, however fast the actors are. The
  stage 4 60M run's log gives the share of time the learner sat waiting for samples.
projected_gain puts them together with Amdahl's law: only the net call's share of an actor's
time shrinks, and the run's gain is capped by the learner.

The engine-only rate (an actor with no net) is already on file:
results/characterization/4p1d_throughput_greedy_<machine>_dev_s10_seed14.json.

Limits: the GPU rate leaves out the cost of sending decisions from actor processes to the GPU
and answers back, so its gain is an upper bound. The split is measured on one process; at 24
processes each actor runs slower (shared CPU cache), which batching may also change.

Run from the repo root (about 5 minutes with the defaults):
    PYTHONPATH=src .venv/bin/python scripts/throughput_ceiling.py --players 4 --seed 14
Output: results/specialist_explore/<config>_throughput_ceiling_<machine>_dev_s<seconds>_seed<s>.json
"""

import argparse
import json
import random
import socket
import time
from pathlib import Path

import torch
from evaluate_agents import DEV_SEED_MIN, REPO_ROOT, RESULTS_ROOT, git_state
from throughput import NET_SEED

from stacker.dmc import EPSILON, batched_choices, choose_action
from stacker.encode import encode_decision
from stacker.game import new_hand, play_out
from stacker.net import QNet, collate_decisions
from stacker.sealed import config_name

# Real decisions kept for the batched timing; more than the largest batch, so a batch never
# holds the same decision twice.
DECISION_POOL_SIZE = 2048
# Batch sizes: one CPU core (an actor's own net call batched over its hands) and the GPU (one
# shared server for every actor).
DEFAULT_CPU_BATCHES = "1,8,32,128"
DEFAULT_GPU_BATCHES = "32,256,1024"
DEFAULT_SPLIT_SECONDS = 60.0
DEFAULT_BATCH_SECONDS = 30.0
# The stage 4 control run (60M samples, 24 actors, teapot): its log holds the learner's wait.
DEFAULT_LEARNER_LOG = (
    REPO_ROOT / "runs" / "specialist_explore" / "4p1d_dmc_teapot_a24_u18750_seed20" / "log.jsonl"
)
DEFAULT_OUT_DIR = RESULTS_ROOT / "specialist_explore"
# The stage's stop rule: a projected run gain under this means a Finding and no build.
BUILD_THRESHOLD = 1.5
MICROSECONDS_PER_SECOND = 1e6


def untrained_net():
    """The control QNet with starting weights fixed by NET_SEED, as every throughput probe uses."""
    torch.manual_seed(NET_SEED)
    return QNet()


def actor_time_split(num_players, seed, seconds):
    """
    Plays self-play hands on one process, one torch thread, for `seconds`, the way a training
    actor does (untrained control net, epsilon EPSILON), and splits the time into buckets.
    num_players: 3 to 8. seed: a dev seed; hand n is dealt from f"{seed}:split:{n}".
    Returns a dict:
      wall_seconds: the whole timed stretch.
      encode_seconds: in encode_decision.
      choose_seconds: in choose_action, the net call included.
      forward_seconds: in the net's forward pass alone; choose minus forward is stacking the
        decision into tensors and the argmax.
      engine_and_other_seconds: the rest: the game engine, play_out, Python overhead.
      decisions, choice_turns (more than one legal action, so the net is called), hands,
      mean_choice_actions (mean K on choice turns).
    Sample storage (one row copied per decision) is left out; a batched actor pays it too.
    """
    torch.set_num_threads(1)
    net = untrained_net()
    seconds_in = {"encode": 0.0, "choose": 0.0, "forward": 0.0}
    counts = {"decisions": 0, "choice_turns": 0, "choice_actions": 0}

    # choose_action only calls q_net(*batch), so a function standing in for the net can time the
    # forward pass without changing stacker.dmc.
    def timed_net(*net_inputs):
        start = time.perf_counter()
        q_values = net(*net_inputs)
        seconds_in["forward"] += time.perf_counter() - start
        return q_values

    # The same two steps as self_play_hand's recording_agent, each timed.
    def timing_agent(observation, actions, agent_rng):
        start = time.perf_counter()
        decision = encode_decision(observation, actions)
        encoded = time.perf_counter()
        index = choose_action(timed_net, decision, EPSILON, agent_rng)
        chosen = time.perf_counter()
        seconds_in["encode"] += encoded - start
        seconds_in["choose"] += chosen - encoded
        counts["decisions"] += 1
        if len(actions) > 1:
            counts["choice_turns"] += 1
            counts["choice_actions"] += len(actions)
        return actions[index]

    hands = 0
    start = time.perf_counter()
    # The check sits at the bottom, so seconds = 0 still plays one hand.
    while True:
        rng = random.Random(f"{seed}:split:{hands}")
        state = new_hand(num_players, rng)
        agents = []
        for _ in state.hands:
            agents.append(timing_agent)
        play_out(state, agents, rng)
        hands += 1
        if time.perf_counter() - start >= seconds:
            break
    wall_seconds = time.perf_counter() - start

    timed_seconds = seconds_in["encode"] + seconds_in["choose"]
    return {
        "wall_seconds": wall_seconds,
        "encode_seconds": seconds_in["encode"],
        "choose_seconds": seconds_in["choose"],
        "forward_seconds": seconds_in["forward"],
        "engine_and_other_seconds": wall_seconds - timed_seconds,
        "decisions": counts["decisions"],
        "choice_turns": counts["choice_turns"],
        "hands": hands,
        "mean_choice_actions": counts["choice_actions"] / counts["choice_turns"],
    }


def choice_decisions(num_players, seed, count):
    """
    Collects `count` real choice-turn decisions (more than one legal action) from self-play with
    the untrained control net at epsilon EPSILON, so the mix of action counts K matches what
    the actors see. Hand n is dealt from f"{seed}:pool:{n}".
    Returns a list of `count` (state, history, mask, action_matrix) tuples from encode_decision.
    """
    net = untrained_net()
    decisions = []

    def collecting_agent(observation, actions, agent_rng):
        decision = encode_decision(observation, actions)
        if len(actions) > 1:
            decisions.append(decision)
        return actions[choose_action(net, decision, EPSILON, agent_rng)]

    hands = 0
    while len(decisions) < count:
        rng = random.Random(f"{seed}:pool:{hands}")
        state = new_hand(num_players, rng)
        agents = []
        for _ in state.hands:
            agents.append(collecting_agent)
        play_out(state, agents, rng)
        hands += 1
    return decisions[:count]


def net_decisions_per_second(net, decisions, batch_size, device, seconds):
    """
    Decisions per second when batch_size decisions share each net call.
    net: a QNet already on `device`. decisions: the pool from choice_decisions; batches take
    consecutive decisions, wrapping at the end. batch_size: B. device: a torch.device.
    seconds: how long to keep calling.
    Returns (decisions per second, net calls made).
    Each call pays stacking, moving to the device, the forward pass and the masked argmax.
    """
    calls = 0
    next_index = 0
    start = time.perf_counter()
    # The check sits at the bottom, so seconds = 0 still makes one call.
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(decisions[next_index % len(decisions)])
            next_index += 1
        batched_choices(net, batch, device)
        calls += 1
        if time.perf_counter() - start >= seconds:
            break
    elapsed = time.perf_counter() - start
    return calls * batch_size / elapsed, calls


def learner_cap(log_path):
    """
    The most any actor speedup can speed a training run, from a run's log.jsonl: each update's
    wall time is wait_seconds (learner idle, waiting for samples) plus learner_seconds. With
    actors infinitely fast the wait is 0, so the run is (wait + learner) / learner times faster.
    Returns that ratio as a float.
    """
    total_wait = 0.0
    total_learner = 0.0
    with open(log_path) as file:
        for line in file:
            entry = json.loads(line)
            # Only update entries carry the timing; others (if any) are skipped.
            if "wait_seconds" in entry:
                total_wait += entry["wait_seconds"]
                total_learner += entry["learner_seconds"]
    return (total_wait + total_learner) / total_learner


def projected_gain(time_split, batched_seconds_per_decision, cap):
    """
    Amdahl's law for batching the net call (only that part of an actor's time changes).
    time_split: the dict actor_time_split returns.
    batched_seconds_per_decision: 1 / net_decisions_per_second at some batch size.
    cap: learner_cap's ratio.
    Returns (actor speedup, run speedup): the actor's decisions per second batched over today,
    and that speedup capped by the learner.

    Today an actor spends wall_seconds / decisions per decision. Batched, it keeps the engine
    and encoding time, and pays the batched cost only on choice turns: a forced turn never
    calls the net, today or batched.
    """
    decisions = time_split["decisions"]
    today_per_decision = time_split["wall_seconds"] / decisions
    kept_seconds = time_split["engine_and_other_seconds"] + time_split["encode_seconds"]
    choice_share = time_split["choice_turns"] / decisions
    batched_per_decision = kept_seconds / decisions + batched_seconds_per_decision * choice_share
    actor_speedup = today_per_decision / batched_per_decision
    if actor_speedup < cap:
        run_speedup = actor_speedup
    else:
        run_speedup = cap
    return actor_speedup, run_speedup


def parse_sizes(text):
    """Splits a comma list of sizes, e.g. "1,8,32", into [1, 8, 32]."""
    sizes = []
    for part in text.split(","):
        sizes.append(int(part))
    return sizes


def main():
    """Measures the split, the batched rates and the cap, projects the gain, writes the JSON."""
    parser = argparse.ArgumentParser(description="Actor-throughput ceiling probe.")
    parser.add_argument("--players", type=int, required=True, help="3 to 8")
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--split-seconds", type=float, default=DEFAULT_SPLIT_SECONDS)
    parser.add_argument("--batch-seconds", type=float, default=DEFAULT_BATCH_SECONDS)
    parser.add_argument("--cpu-batches", default=DEFAULT_CPU_BATCHES)
    parser.add_argument("--gpu-batches", default=DEFAULT_GPU_BATCHES)
    parser.add_argument("--learner-log", type=Path, default=DEFAULT_LEARNER_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Rule 2: seeds 0-2 are test seeds, used only in confirm runs.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )

    # Where the actor's time goes today.
    split = actor_time_split(args.players, args.seed, args.split_seconds)
    print(f"split: {split}")

    # The batched rates. One CPU thread throughout: on the CPU an actor shares the box with 23
    # others; on the GPU only the stacking runs on the CPU.
    decisions = choice_decisions(args.players, args.seed, DECISION_POOL_SIZE)
    cap = learner_cap(args.learner_log)
    settings = []
    for batch_size in parse_sizes(args.cpu_batches):
        settings.append(("cpu", batch_size))
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        for batch_size in parse_sizes(args.gpu_batches):
            settings.append(("cuda", batch_size))

    rows = []
    for device_name, batch_size in settings:
        device = torch.device(device_name)
        net = untrained_net().to(device)
        # One untimed call first: the GPU's first call pays one-time setup.
        batched_choices(net, decisions[:batch_size], device)
        rate, calls = net_decisions_per_second(
            net, decisions, batch_size, device, args.batch_seconds
        )
        actor_speedup, run_speedup = projected_gain(split, 1.0 / rate, cap)
        rows.append(
            {
                "device": device_name,
                "batch_size": batch_size,
                "decisions_per_second": round(rate, 1),
                "calls": calls,
                "actor_speedup": round(actor_speedup, 3),
                "run_speedup": round(run_speedup, 3),
            }
        )
        print(
            f"{device_name} B={batch_size}: {rate:,.0f} decisions/s, actor x{actor_speedup:.2f}, "
            f"run x{run_speedup:.2f}"
        )

    # Consistency: batch 1 on the CPU does what choose_action does, so its cost per decision
    # should be near the split's choose cost per choice turn.
    split_choose_per_turn = split["choose_seconds"] / split["choice_turns"]
    split_choose_microseconds = split_choose_per_turn * MICROSECONDS_PER_SECOND
    print(f"choose per choice turn in the split: {split_choose_microseconds:.0f} us")
    print(f"learner cap: x{cap:.2f}")

    best_run_speedup = 0.0
    for row in rows:
        if row["run_speedup"] > best_run_speedup:
            best_run_speedup = row["run_speedup"]
    if best_run_speedup < BUILD_THRESHOLD:
        verdict = "no build"
    else:
        verdict = "build"
    print(f"best run speedup: x{best_run_speedup:.2f} -> {verdict}")

    commit, uncommitted_changes = git_state()
    machine = socket.gethostname().split(".")[0]
    run_name = (
        f"{config_name(args.players)}_throughput_ceiling_{machine}"
        f"_dev_s{args.split_seconds:g}_seed{args.seed}"
    )
    record = {
        "players": args.players,
        "seed": args.seed,
        "source": "dev",
        "machine": machine,
        "cuda": has_cuda,
        "split_seconds": args.split_seconds,
        "batch_seconds": args.batch_seconds,
        "split": split,
        "split_choose_seconds_per_choice_turn": split_choose_per_turn,
        "learner_log": str(args.learner_log),
        "learner_cap": round(cap, 3),
        "rows": rows,
        "best_run_speedup": best_run_speedup,
        "build_threshold": BUILD_THRESHOLD,
        "verdict": verdict,
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
