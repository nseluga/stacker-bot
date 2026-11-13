"""
Trains the M3 control Q-net by Deep Monte-Carlo self-play on 4p1d (architecture plan §6; stage
plan specialist_explore, stage 3 piece 10).

Starts stacker.dmc.train with --actors actor processes for --updates updates. Exploration only
(rule 2): dev seeds ≥ 10.

Run from the repo root, e.g. 8 actors, 1,000 updates:
    PYTHONPATH=src .venv/bin/python scripts/train_dmc.py --seed 10 --actors 8 --updates 1000
Output, under runs/specialist_explore/<run-name>/ (gitignored):
    config.json   arguments, loop settings, machine, git commit; written before training starts
    log.jsonl     one line per update: loss, gradient norm, wait and learner seconds, samples/s
    checkpoint_<updates done>.pt   the learner's weights (a QNet state_dict)
The default run name is <config>_dmc_<machine>_a<actors>_u<updates>_seed<seed>.
"""

import argparse
import json
import os
import platform
import socket

import torch
from evaluate_agents import DEV_SEED_MIN, REPO_ROOT, git_state

from stacker.dmc import (
    CHECKPOINT_EVERY_UPDATES,
    EPSILON,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_PLAYERS,
    QUEUE_MAX_HANDS,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    SAMPLES_PER_UPDATE,
    train,
)
from stacker.sealed import config_name

# Every M3 training run lives under the milestone's slug (manifest: runs/<slug>/<run-name>/).
RUNS_DIR = REPO_ROOT / "runs" / "specialist_explore"


def main():
    """Parses the command line, writes the run's config.json, trains, prints a summary line."""
    parser = argparse.ArgumentParser(description="Deep Monte-Carlo self-play training on 4p1d.")
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--actors", type=int, required=True, help="actor processes")
    parser.add_argument("--updates", type=int, required=True, help="learner updates")
    parser.add_argument("--samples-per-update", type=int, default=SAMPLES_PER_UPDATE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY_UPDATES)
    parser.add_argument("--run-name", help="directory name under runs/specialist_explore/")
    args = parser.parse_args()

    # Rule 2: seeds 0-2 are test seeds, used only in confirm runs; M3 is exploration.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )

    # The short hostname names the machine, so laptop and teapot runs never collide.
    machine = socket.gethostname().split(".")[0]
    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"{config_name(NUM_PLAYERS)}_dmc_{machine}_a{args.actors}_u{args.updates}"
            f"_seed{args.seed}"
        )

    # exist_ok=False: a directory that is already there is an earlier run, and it is kept.
    run_dir = RUNS_DIR / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"{run_dir} exists; pick another --run-name or remove the old run")

    commit, uncommitted_changes = git_state()

    # CPUs this process may run on: an account limit can be below cpu_count (teapot: 39 of 48).
    # sched_getaffinity is Linux only, so macOS records None.
    if hasattr(os, "sched_getaffinity"):
        usable_cpus = len(os.sched_getaffinity(0))
    else:
        usable_cpus = None

    # Written before training, so a run that dies still says what it was.
    config = {
        "seed": args.seed,
        "actors": args.actors,
        "updates": args.updates,
        "samples_per_update": args.samples_per_update,
        "checkpoint_every": args.checkpoint_every,
        "source": "dev",
        "players": NUM_PLAYERS,
        "epsilon": EPSILON,
        "learning_rate": LEARNING_RATE,
        "rmsprop_alpha": RMSPROP_ALPHA,
        "rmsprop_eps": RMSPROP_EPS,
        "rmsprop_momentum": RMSPROP_MOMENTUM,
        "max_grad_norm": MAX_GRAD_NORM,
        "queue_max_hands": QUEUE_MAX_HANDS,
        "machine": machine,
        "cpu_count": os.cpu_count(),
        "usable_cpus": usable_cpus,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
    }
    with open(run_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    train(
        args.seed,
        args.actors,
        run_dir,
        args.updates,
        samples_per_update=args.samples_per_update,
        checkpoint_every=args.checkpoint_every,
    )

    # Summary from the log train wrote. The first update's time includes starting the actor
    # processes, so the mean rate leaves it out.
    last_loss = None
    later_rates = []
    with open(run_dir / "log.jsonl") as log_file:
        for line in log_file:
            log_line = json.loads(line)
            last_loss = log_line["loss"]
            if log_line["update"] > 1:
                later_rates.append(log_line["samples_per_second"])
    summary = f"{run_name}: {args.updates} updates, last loss {last_loss:.4f}"
    if later_rates:
        mean_rate = sum(later_rates) / len(later_rates)
        summary += f", {mean_rate:,.0f} samples/s after the first update"
    print(summary)


# The guard is required, not style: each actor process starts by importing this file again, and
# without it every actor would start a training run of its own.
if __name__ == "__main__":
    main()
