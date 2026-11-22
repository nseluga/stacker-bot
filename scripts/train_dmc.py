"""
Trains the M3 control Q-net by Deep Monte-Carlo self-play on 4p1d (architecture plan §6; stage
plan specialist_explore, stage 3 piece 10).

Starts stacker.dmc.train with --actors actor processes for --updates updates. Exploration only
(rule 2): dev seeds ≥ 10.

Run from the repo root, e.g. 8 actors, 1,000 updates:
    PYTHONPATH=src .venv/bin/python scripts/train_dmc.py --seed 10 --actors 8 --updates 1000
On a machine with a GPU, the learner on it and 4 CPU threads left to the learner:
    ... --device cuda --learner-threads 4
The label-shuffle control (stage 4, spec part 13), the same run with the returns shuffled:
    ... --shuffle-returns
An exploration arm (stage 5, spec parts 16-17): any of --width, --layers, --history-hidden,
--history-window, --lr, --epsilon, --optimizer, each defaulting to the control, plus --arm, a tag for the run:
    ... --width 256 --lr 3e-4 --arm width256-lr3e-4 --base-run <the base run's name>
An arm needs --base-run: before anything is made, its config is compared with the base run's
config.json, and the run is refused if a setting differs that no flag on the command line
named (check_against_base).
Output, under runs/specialist_explore/<run-name>/ (gitignored):
    config.json   arguments, loop settings, machine, git commit; written before training starts
    log.jsonl     one line per update: samples so far, loss, the variance of the batch's
                  returns, gradient norm, wait and learner seconds, samples/s
    checkpoint_<updates done>.pt   the learner's weights (a QNet state_dict, held on the CPU);
                  checkpoint_000000.pt is the untrained net
The default run name is <config>_dmc_<machine>_a<actors>_u<updates>_seed<seed>, with _shuffled
at the end for the label-shuffle control and _<arm> at the end of an arm. A name given with
--run-name is used as it is.
"""

import argparse
import json
import os
import platform
import re
import socket
import sys

import torch
from evaluate_agents import DEV_SEED_MIN, REPO_ROOT, git_state

from stacker.dmc import (
    ADAM_BETAS,
    ADAM_EPS,
    CHECKPOINT_EVERY_UPDATES,
    CONTROL_OPTIMIZER,
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
from stacker.encode import HISTORY_WINDOW
from stacker.net import DEFAULT_HIDDEN_LAYERS, DEFAULT_HISTORY_HIDDEN, DEFAULT_MLP_WIDTH
from stacker.sealed import config_name

# Every M3 training run lives under the milestone's slug (manifest: runs/<slug>/<run-name>/).
RUNS_DIR = REPO_ROOT / "runs" / "specialist_explore"
# An arm tag goes into a directory name: lowercase letters, digits and dashes only.
ARM_TAG_PATTERN = r"[a-z0-9-]+"
# The optimizers build_optimizer knows; argparse refuses any other name before anything starts.
OPTIMIZER_CHOICES = ("rmsprop", "adam")
# config.json keys that describe the machine and the code version, not the experiment. They may
# differ from the base run's without being named (decision: Nate, 2026-11-22, piece 3b).
BOOKKEEPING_KEYS = (
    "machine",
    "cpu_count",
    "usable_cpus",
    "gpu_name",
    "learner_threads",
    "python",
    "torch",
    "git_commit",
    "uncommitted_changes",
)
# The flags whose config.json key is not the flag's own name; every other flag's key is its
# argparse name (--samples-per-update -> samples_per_update).
CONFIG_KEY_OF_FLAG = {"layers": "hidden_layers", "lr": "learning_rate"}


def check_against_base(config, base_config, named_keys):
    """
    Refuses an arm whose settings differ from its base run's in a key it did not name (stage
    plan, stage 5 check "config").
    config: the new run's config dict, as it will be written to config.json.
    base_config: the base run's config.json, loaded.
    named_keys: set of config keys whose flags are on the command line.
    Returns None when every differing key, bookkeeping aside, is named.
    Raises SystemExit naming each key that differs and was not named, with both values.
    A key that only one of the two configs has counts as differing (its value there is None).
    """
    all_keys = set(config) | set(base_config)
    unnamed = []
    for key in sorted(all_keys):
        if key in BOOKKEEPING_KEYS or key in named_keys:
            continue
        if config.get(key) != base_config.get(key):
            unnamed.append(f"{key}: base {base_config.get(key)!r}, this run {config.get(key)!r}")
    if unnamed:
        raise SystemExit(
            "differs from the base run without a flag naming it: " + "; ".join(unnamed)
        )


def main():
    """Parses the command line, writes the run's config.json, trains, prints a summary line."""
    parser = argparse.ArgumentParser(description="Deep Monte-Carlo self-play training on 4p1d.")
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--actors", type=int, required=True, help="actor processes")
    parser.add_argument("--updates", type=int, required=True, help="learner updates")
    parser.add_argument("--samples-per-update", type=int, default=SAMPLES_PER_UPDATE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY_UPDATES)
    parser.add_argument("--run-name", help="directory name under runs/specialist_explore/")
    parser.add_argument("--device", default="cpu", help="where the learner runs: cpu, cuda, cuda:1")
    parser.add_argument("--learner-threads", type=int, help="CPU threads of the learner process")
    parser.add_argument(
        "--shuffle-returns",
        action="store_true",
        help="label-shuffle control: shuffle the returns inside every batch",
    )
    # The arm settings (stage 5). Each default is the control's value.
    parser.add_argument("--width", type=int, default=DEFAULT_MLP_WIDTH, help="MLP width")
    parser.add_argument(
        "--layers", type=int, default=DEFAULT_HIDDEN_LAYERS, help="hidden MLP layers"
    )
    parser.add_argument(
        "--history-hidden", type=int, default=DEFAULT_HISTORY_HIDDEN, help="LSTM summary size"
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=HISTORY_WINDOW,
        help="moves of history the net reads; 0 = none",
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON, help="actors' random-action probability"
    )
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default=CONTROL_OPTIMIZER)
    parser.add_argument("--arm", help="arm tag, added to the run name: a-z, 0-9 and -")
    parser.add_argument(
        "--base-run", help="the run this arm is compared with, a name under runs/specialist_explore/"
    )
    args = parser.parse_args()

    # Rule 2: seeds 0-2 are test seeds, used only in confirm runs; M3 is exploration.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )

    # The tag becomes part of a directory name, so anything but plain characters is refused.
    if args.arm is not None and re.fullmatch(ARM_TAG_PATTERN, args.arm) is None:
        raise SystemExit(f"--arm takes a-z, 0-9 and - only, got {args.arm!r}")
    if args.arm is not None and args.base_run is None:
        raise SystemExit("--arm needs --base-run: the run its config is checked against")

    # Checked here, before the run's directory is made, so a wrong device leaves nothing behind.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device}: torch finds no CUDA GPU on this machine")

    # With the learner on a GPU its CPU threads have little to do, but each one still takes a
    # core from the actors. The setting is this process's only: an actor is its own process.
    if args.learner_threads is not None:
        torch.set_num_threads(args.learner_threads)

    # The short hostname names the machine, so laptop and teapot runs never collide.
    machine = socket.gethostname().split(".")[0]
    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"{config_name(NUM_PLAYERS)}_dmc_{machine}_a{args.actors}_u{args.updates}"
            f"_seed{args.seed}"
        )
        # The control has the same actors, updates and seed as the run it is compared with, so
        # without a mark of its own the two would ask for one directory.
        if args.shuffle_returns:
            run_name += "_shuffled"
        if args.arm is not None:
            run_name += f"_{args.arm}"

    commit, uncommitted_changes = git_state()

    # CPUs this process may run on: can be below cpu_count (teapot: 39 of 48, a quota shared by
    # all users, so other jobs take from it).
    # sched_getaffinity is Linux only, so macOS records None.
    if hasattr(os, "sched_getaffinity"):
        usable_cpus = len(os.sched_getaffinity(0))
    else:
        usable_cpus = None

    # The GPU's model name, e.g. "NVIDIA GeForce RTX 4090"; None when the learner is on the CPU.
    gpu_name = None
    if args.device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(args.device)

    # Written before training, so a run that dies still says what it was.
    config = {
        "seed": args.seed,
        "actors": args.actors,
        "updates": args.updates,
        "samples_per_update": args.samples_per_update,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "gpu_name": gpu_name,
        # What torch reports, so a run without --learner-threads still records a number.
        "learner_threads": torch.get_num_threads(),
        "shuffle_returns": args.shuffle_returns,
        "source": "dev",
        "players": NUM_PLAYERS,
        "arm": args.arm,
        "base_run": args.base_run,
        "epsilon": args.epsilon,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "width": args.width,
        "hidden_layers": args.layers,
        "history_hidden": args.history_hidden,
        "history_window": args.history_window,
        # Both optimizers' fixed settings are recorded in every run, so the two kinds of
        # config have the same keys and differ only in "optimizer".
        "rmsprop_alpha": RMSPROP_ALPHA,
        "rmsprop_eps": RMSPROP_EPS,
        "rmsprop_momentum": RMSPROP_MOMENTUM,
        # A list: JSON has no tuples.
        "adam_betas": list(ADAM_BETAS),
        "adam_eps": ADAM_EPS,
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
    if args.base_run is not None:
        base_config_path = RUNS_DIR / args.base_run / "config.json"
        if not base_config_path.exists():
            raise SystemExit(f"--base-run {args.base_run}: no {base_config_path}")
        with open(base_config_path) as file:
            base_config = json.load(file)
        # A flag is named when it is written on the command line, even at its default value:
        # "--lr 3e-4" and "--lr=3e-4" both name learning_rate.
        named_keys = set()
        for action in parser._actions:
            for option in action.option_strings:
                for word in sys.argv[1:]:
                    if word == option or word.startswith(option + "="):
                        named_keys.add(CONFIG_KEY_OF_FLAG.get(action.dest, action.dest))
        check_against_base(config, base_config, named_keys)

    # Made only now, so a refused run leaves no directory behind.
    # exist_ok=False: a directory that is already there is an earlier run, and it is kept.
    run_dir = RUNS_DIR / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"{run_dir} exists; pick another --run-name or remove the old run")
    with open(run_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    # Every setting is read back from the config just written, so the record and the run cannot
    # disagree.
    train(
        config["seed"],
        config["actors"],
        run_dir,
        config["updates"],
        samples_per_update=config["samples_per_update"],
        checkpoint_every=config["checkpoint_every"],
        device=config["device"],
        shuffle_control=config["shuffle_returns"],
        learning_rate=config["learning_rate"],
        epsilon=config["epsilon"],
        optimizer=config["optimizer"],
        width=config["width"],
        hidden_layers=config["hidden_layers"],
        history_hidden=config["history_hidden"],
        history_window=config["history_window"],
    )

    # Summary from the log train wrote. The first update's time includes starting the actor
    # processes, so the mean rate leaves it out.
    last_loss = None
    last_return_variance = None
    later_rates = []
    with open(run_dir / "log.jsonl") as log_file:
        for line in log_file:
            log_line = json.loads(line)
            last_loss = log_line["loss"]
            last_return_variance = log_line["return_variance"]
            if log_line["update"] > 1:
                later_rates.append(log_line["samples_per_second"])
    # The return variance is the loss of a net that has learned nothing, so the two are read
    # side by side.
    summary = (
        f"{run_name}: {args.updates} updates, last loss {last_loss:.4f}"
        f" (return variance {last_return_variance:.4f})"
    )
    if later_rates:
        mean_rate = sum(later_rates) / len(later_rates)
        summary += f", {mean_rate:,.0f} samples/s after the first update"
    print(summary)


# The guard is required, not style: each actor process starts by importing this file again, and
# without it every actor would start a training run of its own.
if __name__ == "__main__":
    main()
