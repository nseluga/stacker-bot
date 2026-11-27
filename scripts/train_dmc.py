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
The classifier head (set 9, spec part 18): the net predicts the chance of each placement and
learns from smoothed placement labels instead of the score itself:
    ... --head classifier --arm head-classifier --base-run <the base run's name>
Sample reuse (set 12): after each fresh batch, one extra update on a batch drawn from the last
1M samples trained on (--reuse-pool-samples changes the pool's size):
    ... --reuse-updates 1 --arm reuse1 --base-run <the base run's name>
The fast actor (design C, decision log 2026-11-25): each actor plays many hands at once on its
own net on a GPU, refreshed from the learner every update:
    ... --fast-actor --actors 16 --actor-hands 64 --actor-devices cuda:0,cuda:1 --device cuda:0
The heuristic chair (set 8): one random seat of every hand is played by the heuristic, and that
seat's decisions are not trained on; today's actor only, so it is refused with --fast-actor:
    ... --heuristic-chair --arm heuristic-chair --base-run <the base run's name>
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
at the end for the label-shuffle control, _fast for a fast-actor run and _<arm> at the end of an
arm. A name given with
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
    CONTROL_REUSE_UPDATES,
    EPSILON,
    FAST_ACTOR_HANDS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_PLAYERS,
    QUEUE_MAX_HANDS,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    REUSE_POOL_SAMPLES,
    SAMPLES_PER_UPDATE,
    train,
)
from stacker.encode import HISTORY_WINDOW
from stacker.net import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HISTORY_HIDDEN,
    DEFAULT_MLP_WIDTH,
    HEAD_CLASSIFIER,
    HEAD_MSE,
)
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
# Where fast actors play when --actor-devices is not given.
DEFAULT_ACTOR_DEVICES = "cpu"
# The config.json keys a fast-actor run adds. Writing --fast-actor names all three for the base
# run check, so the defaults of the other two are not refused as unnamed differences.
FAST_ACTOR_KEYS = ("fast_actor", "actor_hands", "actor_devices")

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


def fast_actor_settings(args):
    """
    The fast-actor part of a run's config, and the config keys it names for check_against_base.
    args: the parsed command line, with fast_actor (bool), actor_hands (int or None) and
      actor_devices (a comma-separated str, or None) from --fast-actor, --actor-hands and
      --actor-devices.
    Returns (config_entries, named_keys). A run without --fast-actor gets ({}, set()): its
      config has none of the keys, so it matches a base run made before the fast actor existed.
      A fast run gets all three keys, actor_devices as a list, and all three named.
    Raises SystemExit when --actor-hands or --actor-devices is given without --fast-actor:
      train would ignore them, and the run would not be the one asked for.
    """
    if not args.fast_actor:
        if args.actor_hands is not None or args.actor_devices is not None:
            raise SystemExit("--actor-hands and --actor-devices need --fast-actor")
        return {}, set()

    actor_hands = args.actor_hands
    if actor_hands is None:
        actor_hands = FAST_ACTOR_HANDS
    actor_devices = args.actor_devices
    if actor_devices is None:
        actor_devices = DEFAULT_ACTOR_DEVICES
    config_entries = {
        "fast_actor": True,
        "actor_hands": actor_hands,
        "actor_devices": actor_devices.split(","),
    }
    return config_entries, set(FAST_ACTOR_KEYS)


def fast_actor_settings_use_cuda(fast_entries):
    """True when a fast run's actor devices include a GPU."""
    for device_name in fast_entries.get("actor_devices", []):
        if device_name.startswith("cuda"):
            return True
    return False


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
    parser.add_argument(
        "--head", choices=(HEAD_MSE, HEAD_CLASSIFIER), default=HEAD_MSE, help="net output head"
    )
    # The fast actor. The two settings default to None so a value given without --fast-actor
    # can be told apart from no value (fast_actor_settings).
    parser.add_argument(
        "--fast-actor", action="store_true", help="design C actors: own net, many hands"
    )
    parser.add_argument(
        "--actor-hands", type=int, help=f"hands per fast actor (default {FAST_ACTOR_HANDS})"
    )
    parser.add_argument(
        "--actor-devices",
        help=f"fast actors' devices, comma-separated (default {DEFAULT_ACTOR_DEVICES})",
    )
    parser.add_argument(
        "--heuristic-chair",
        action="store_true",
        help="one random seat per hand played by the heuristic (set 8)",
    )
    # The pool size defaults to None so a size given without --reuse-updates can be told apart
    # from no size, and refused.
    parser.add_argument(
        "--reuse-updates",
        type=int,
        default=CONTROL_REUSE_UPDATES,
        help="extra updates per fresh batch on samples drawn from the reuse pool (set 12)",
    )
    parser.add_argument(
        "--reuse-pool-samples",
        type=int,
        help=f"samples the reuse pool holds (default {REUSE_POOL_SAMPLES})",
    )
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
    fast_entries, fast_named_keys = fast_actor_settings(args)
    # train refuses this pair too, but only after the run's directory and config exist.
    # train would ignore a pool size with no extra updates, and the run would not be the one asked.
    if args.reuse_pool_samples is not None and args.reuse_updates == 0:
        raise SystemExit("--reuse-pool-samples needs --reuse-updates")
    if args.heuristic_chair and args.fast_actor:
        raise SystemExit("--heuristic-chair runs on today's actor only; drop --fast-actor")
    if fast_actor_settings_use_cuda(fast_entries) and not torch.cuda.is_available():
        raise SystemExit(
            "--actor-devices names a GPU: torch finds no CUDA GPU on this machine"
        )

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
        if args.fast_actor:
            run_name += "_fast"
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
    # Only a fast run has these keys (fast_actor_settings).
    config.update(fast_entries)
    # Only a chair run has this key, so every other arm's config keeps the base run's keys.
    # Writing --heuristic-chair names it for check_against_base below.
    if args.heuristic_chair:
        config["heuristic_chair"] = True
    # Only a classifier run has this key, so a control config stays as it was before the head
    # existed and still matches the base runs made then.
    if args.head == HEAD_CLASSIFIER:
        config["head"] = HEAD_CLASSIFIER
    # Only a reuse run has these keys, so a control config still matches the base runs. The pool
    # size is part of the arm --reuse-updates names, so it is named with it.
    reuse_named_keys = set()
    if args.reuse_updates > 0:
        config["reuse_updates"] = args.reuse_updates
        config["reuse_pool_samples"] = REUSE_POOL_SAMPLES
        if args.reuse_pool_samples is not None:
            config["reuse_pool_samples"] = args.reuse_pool_samples
        reuse_named_keys = {"reuse_updates", "reuse_pool_samples"}
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
        named_keys |= fast_named_keys
        named_keys |= reuse_named_keys
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
        # A plain run's config has none of the fast keys, so train's defaults apply.
        fast_actor=config.get("fast_actor", False),
        actor_hands=config.get("actor_hands", FAST_ACTOR_HANDS),
        actor_devices=tuple(config.get("actor_devices", [DEFAULT_ACTOR_DEVICES])),
        # Only a chair run's config has the key.
        heuristic_chair=config.get("heuristic_chair", False),
        # A config without the key is an mse run.
        head=config.get("head", HEAD_MSE),
        # Only a reuse run's config has the keys; without them train makes no extra updates.
        reuse_updates=config.get("reuse_updates", CONTROL_REUSE_UPDATES),
        reuse_pool_samples=config.get("reuse_pool_samples", REUSE_POOL_SAMPLES),
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
