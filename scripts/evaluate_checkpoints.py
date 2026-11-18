"""
Learning curve of one training run: each saved checkpoint played against fixed field agents on
dev deals (stage plan specialist_explore, stage 4 piece 8; spec part 14, equation 15).

Exploration only (rule 2): dev seeds ≥ 10. There is no path to the sealed deals here.

Run from the repo root, e.g. every checkpoint of a run against random and greedy:
    PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoints.py --run <run-name>
Reads runs/specialist_explore/<run-name>/ (config.json and checkpoint_*.pt). Writes, under
results/specialist_explore/, three files named <run-name>_eval_dev_n<deals>_seed<seed>:
    .csv         the curve, one row per (checkpoint, field): checkpoint (updates done), samples
                 (training samples seen), field, mean, ci_low, ci_high, standard_deviation,
                 seconds
    _deals.csv   one row per (checkpoint, field, deal): the paired difference of that deal.
                 Every checkpoint plays the same deals, so two can be compared deal by deal.
    .json        the arguments, git state now, and the training run's own commit
A rerun with the same run, deals and seed replaces the three files. With --checkpoints the
name ends _ckpt<numbers>, so scoring a few checkpoints never replaces the full curve.
"""

import argparse
import csv
import functools
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from evaluate_agents import AGENTS, DEV_SEED_MIN, RESULTS_ROOT, git_state, select_deals
from train_dmc import RUNS_DIR

from stacker.dmc import choose_action
from stacker.encode import encode_decision
from stacker.evaluate import bootstrap_ci, paired_differences
from stacker.net import QNet

# Evaluation plays the net's best action on every turn: no random moves (equation 15).
EVAL_EPSILON = 0.0

# The format: the checkpoint holds seat 0 at rotation 0 and the field agent the other seats
# (1v3 at four players). paired_differences plays every rotation of it.
TEST_SEATS = (0,)

# Results sit under the same slug as the runs they score.
RESULTS_DIR = RESULTS_ROOT / RUNS_DIR.name


def load_checkpoint_net(path):
    """
    Builds the net a checkpoint file holds, ready to play.
    path: a checkpoint_<updates done>.pt written by stacker.dmc.train (a QNet state_dict).
    Returns a QNet on the CPU, in eval mode.
    """
    net = QNet()
    # map_location: a checkpoint is saved from the CPU copy, but loading must not depend on it.
    # weights_only: the file holds tensors only, so torch refuses to run any code stored in it.
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    # Eval mode turns off anything that behaves differently in training (dropout, batch
    # statistics). QNet has none today; this keeps evaluation right if one is added.
    net.eval()
    return net


def checkpoint_agent(net, observation, actions, rng):
    """
    Plays the action the net scores highest (equation 15).
    net: a QNet from load_checkpoint_net. The caller binds it with functools.partial, since
    play_out calls an agent as agent(observation, actions, rng).
    observation, actions, rng: as play_out passes them. Returns one of actions.
    A forced turn (one legal action) returns it without calling the net; equal scores go to the
    lowest index; no gradients are recorded. All three are choose_action's own behaviour.
    """
    decision = encode_decision(observation, actions)
    # With epsilon 0 choose_action still draws one number from rng on a turn with a choice, and
    # then never takes the random branch. The draw is the same on every replay of a deal.
    index = choose_action(net, decision, EVAL_EPSILON, rng)
    return actions[index]


def evaluate_checkpoint(task):
    """
    Scores one checkpoint against one field agent on dev deals. Runs in a worker process.
    task: (checkpoint_path, field_name, num_players, num_deals, seed), one tuple because a
    process pool hands a worker one argument. field_name is a key of AGENTS; seed is ≥ 10.
    Returns a dict: checkpoint (file name), field, mean, ci_low, ci_high (95% bootstrap CI of
    the mean), standard_deviation, seconds (play time), differences (list of num_deals floats,
    the per-deal paired differences in deal order). Positive means the checkpoint is stronger.

    Every checkpoint sees the same games: the deals come from seed alone, and each rotation's
    rng is named by seed, deal and rotation, never by the checkpoint. So two checkpoints'
    differences can be compared deal by deal.
    """
    checkpoint_path, field_name, num_players, num_deals, seed = task

    # Many workers run at once, one per core. Torch's default would give each of them a thread
    # per core, all waiting on each other.
    torch.set_num_threads(1)

    net = load_checkpoint_net(checkpoint_path)
    test_agent = functools.partial(checkpoint_agent, net)

    # False: dev deals. It is fixed here, so no argument can reach the sealed set, and
    # select_deals raises on a seed below 10.
    deck_orders = select_deals(num_players, num_deals, seed, False)

    start = time.perf_counter()
    differences = paired_differences(
        deck_orders, num_players, test_agent, AGENTS[field_name], TEST_SEATS, seed
    )
    seconds = time.perf_counter() - start

    # Same rng name as evaluate_agents.py, so the CI replays from the seed.
    mean, low, high = bootstrap_ci(differences, random.Random(f"{seed}:bootstrap"))

    # A standard deviation needs two values.
    standard_deviation = None
    if len(differences) > 1:
        standard_deviation = statistics.stdev(differences)

    return {
        "checkpoint": Path(checkpoint_path).name,
        "field": field_name,
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "standard_deviation": standard_deviation,
        "seconds": round(seconds, 2),
        "differences": differences,
    }


def main():
    """Parses the command line, scores the run's checkpoints, and writes the three files."""
    parser = argparse.ArgumentParser(description="Score a run's checkpoints on dev deals.")
    parser.add_argument("--run", required=True, help="run name under runs/specialist_explore/")
    parser.add_argument("--fields", default="random,greedy", help="field agents, comma-separated")
    parser.add_argument("--deals", type=int, default=1000, help="deals per (checkpoint, field)")
    parser.add_argument("--seed", type=int, default=22, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--checkpoints", help="updates-done numbers, e.g. 0,3125; default all")
    parser.add_argument("--workers", type=int, help="processes; default the usable cores")
    args = parser.parse_args()

    # select_deals refuses the seed too, but inside a worker, after the pool has started.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(f"dev seeds are ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}")

    field_names = args.fields.split(",")
    for field_name in field_names:
        if field_name not in AGENTS:
            raise SystemExit(f"unknown field {field_name!r}; choose from {sorted(AGENTS)}")

    run_dir = RUNS_DIR / args.run
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"{config_path} not found: not a training run's directory")
    with open(config_path) as file:
        run_config = json.load(file)

    # checkpoint_<updates done>.pt, zero-padded, so sorting the names sorts by updates done.
    updates_by_path = {}
    for path in sorted(run_dir.glob("checkpoint_*.pt")):
        updates_by_path[path] = int(path.stem.split("_")[1])

    if args.checkpoints is not None:
        wanted = set()
        for part in args.checkpoints.split(","):
            wanted.add(int(part))
        missing = wanted - set(updates_by_path.values())
        if missing:
            raise SystemExit(f"{run_dir} has no checkpoint at updates {sorted(missing)}")
        chosen = {}
        for path, updates in updates_by_path.items():
            if updates in wanted:
                chosen[path] = updates
        updates_by_path = chosen
    if len(updates_by_path) == 0:
        raise SystemExit(f"{run_dir} holds no checkpoint_*.pt")

    # One task per (checkpoint, field), checkpoints in training order.
    tasks = []
    for path in updates_by_path:
        for field_name in field_names:
            tasks.append((str(path), field_name, run_config["players"], args.deals, args.seed))

    # sched_getaffinity (Linux only) counts the cores this process may use, which on a shared
    # machine can be fewer than the machine has.
    workers = args.workers
    if workers is None:
        if hasattr(os, "sched_getaffinity"):
            workers = len(os.sched_getaffinity(0))
        else:
            workers = os.cpu_count()
    workers = min(workers, len(tasks))

    # "spawn", as in train: each worker is a fresh Python process, so none inherits torch's
    # threads. map returns the results in task order, whichever task finishes first.
    start = time.perf_counter()
    context = torch.multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        results = pool.map(evaluate_checkpoint, tasks)
    total_seconds = time.perf_counter() - start

    name = f"{args.run}_eval_dev_n{args.deals}_seed{args.seed}"
    # A partial curve gets its own name, so it cannot replace the full one.
    if args.checkpoints is not None:
        chosen_numbers = []
        for updates in updates_by_path.values():
            chosen_numbers.append(str(updates))
        name += "_ckpt" + "-".join(chosen_numbers)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / (name + ".csv"), "w", newline="") as curve_file:
        with open(RESULTS_DIR / (name + "_deals.csv"), "w", newline="") as deals_file:
            curve = csv.writer(curve_file)
            deals = csv.writer(deals_file)
            curve.writerow(
                ["checkpoint", "samples", "field", "mean", "ci_low", "ci_high",
                 "standard_deviation", "seconds"]
            )
            deals.writerow(["checkpoint", "field", "deal_index", "difference"])
            for task, result in zip(tasks, results):
                updates = updates_by_path[Path(task[0])]
                # Training samples the checkpoint has seen: the curve's x-axis.
                samples = updates * run_config["samples_per_update"]
                curve.writerow(
                    [updates, samples, result["field"], result["mean"], result["ci_low"],
                     result["ci_high"], result["standard_deviation"], result["seconds"]]
                )
                for deal_index, difference in enumerate(result["differences"]):
                    deals.writerow([updates, result["field"], deal_index, difference])
                print(
                    f"{samples:>12,} samples vs {result['field']}: mean {result['mean']:+.4f}, "
                    f"95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
                )

    commit, uncommitted_changes = git_state()
    summary = {
        "run": args.run,
        "fields": field_names,
        "deals": args.deals,
        "seed": args.seed,
        "source": "dev",
        "test_seats": list(TEST_SEATS),
        "checkpoints": list(updates_by_path.values()),
        "workers": workers,
        "total_seconds": round(total_seconds, 2),
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
        # The commit the training run recorded: the code that made the checkpoints.
        "run_git_commit": run_config["git_commit"],
        "run_uncommitted_changes": run_config["uncommitted_changes"],
    }
    with open(RESULTS_DIR / (name + ".json"), "w") as file:
        json.dump(summary, file, indent=2)
    print(f"{name}: {len(tasks)} evaluations in {total_seconds:.1f} s on {workers} workers")


if __name__ == "__main__":
    main()
