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
from stacker.encode import HISTORY_WINDOW, encode_decision
from stacker.evaluate import bootstrap_ci, mixed_table_differences, paired_differences
from stacker.net import DEFAULT_HIDDEN_LAYERS, DEFAULT_HISTORY_HIDDEN, DEFAULT_MLP_WIDTH, QNet

# Evaluation plays the net's best action on every turn: no random moves (equation 15).
EVAL_EPSILON = 0.0

# The format: the checkpoint holds seat 0 at rotation 0 and the field agent the other seats
# (1v3 at four players). paired_differences plays every rotation of it.
TEST_SEATS = (0,)

# The config.json keys that set the net's sizes and the history it reads (stage 5). A run's
# checkpoint is scored with the net and window its own config names.
NET_CONFIG_KEYS = ("width", "hidden_layers", "history_hidden", "history_window")
# Configs written before stage 5 have none of those keys: every such run was the control, since
# the settings did not exist yet (decision: Nate, 2026-11-22, piece 5, option C).
CONTROL_NET_CONFIG = {
    "width": DEFAULT_MLP_WIDTH,
    "hidden_layers": DEFAULT_HIDDEN_LAYERS,
    "history_hidden": DEFAULT_HISTORY_HIDDEN,
    "history_window": HISTORY_WINDOW,
}

# The mixed table (stage 5, "Mixed table"): the field seats hold one of each style, the field
# run's net and these agents. Its result is labelled MIXED_FIELD.
MIXED_FIELD = "mixed"
MIXED_FIELD_AGENTS = ("heuristic", "greedy")

# Results sit under the same slug as the runs they score.
RESULTS_DIR = RESULTS_ROOT / RUNS_DIR.name


def load_checkpoint_net(path, config):
    """
    Builds the net a checkpoint file holds, at the sizes its run used, ready to play.
    path: a checkpoint_<updates done>.pt written by stacker.dmc.train (a QNet state_dict).
    config: the run's config.json, loaded.
    Returns (net, history_window): a QNet on the CPU, in eval mode, and the number of history
      moves its decisions must be encoded with.
    A config with none of NET_CONFIG_KEYS is from before stage 5 and gets the control's values.
    Raises ValueError for a config with some of the keys but not all: a stage 5 config always
    has every one, so a gap is a broken record, never filled in.
    """
    missing = []
    for key in NET_CONFIG_KEYS:
        if key not in config:
            missing.append(key)
    if len(missing) == len(NET_CONFIG_KEYS):
        net_config = CONTROL_NET_CONFIG
    elif missing:
        raise ValueError(f"run config lacks {missing}; it has the other stage 5 net keys")
    else:
        net_config = config

    # A size that does not match the saved weights makes load_state_dict raise, so a wrong size
    # fails loudly. A wrong window would not (the LSTM reads any number of rows), which is why
    # the window is returned from the same config instead of left to a default.
    net = QNet(
        width=net_config["width"],
        hidden_layers=net_config["hidden_layers"],
        history_hidden=net_config["history_hidden"],
    )
    # map_location: a checkpoint is saved from the CPU copy, but loading must not depend on it.
    # weights_only: the file holds tensors only, so torch refuses to run any code stored in it.
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    # Eval mode turns off anything that behaves differently in training (dropout, batch
    # statistics). QNet has none today; this keeps evaluation right if one is added.
    net.eval()
    return net, net_config["history_window"]


def checkpoint_agent(net, history_window, observation, actions, rng):
    """
    Plays the action the net scores highest (equation 15).
    net, history_window: as load_checkpoint_net returns them. The caller binds both with
    functools.partial, since play_out calls an agent as agent(observation, actions, rng).
    observation, actions, rng: as play_out passes them. Returns one of actions.
    A forced turn (one legal action) returns it without calling the net; equal scores go to the
    lowest index; no gradients are recorded. All three are choose_action's own behaviour.
    """
    decision = encode_decision(observation, actions, history_window)
    # With epsilon 0 choose_action still draws one number from rng on a turn with a choice, and
    # then never takes the random branch. The draw is the same on every replay of a deal.
    index = choose_action(net, decision, EVAL_EPSILON, rng)
    return actions[index]


def evaluate_checkpoint(task):
    """
    Scores one checkpoint against one field agent on dev deals. Runs in a worker process.
    task: (checkpoint_path, field_name, num_players, num_deals, seed, run_config,
    field_checkpoint), one tuple because a process pool hands a worker one argument. seed is
    ≥ 10; run_config is the run's config.json, which names the net's sizes and window.
    field_checkpoint: None, and field_name is a key of AGENTS; or (path, config) of a second
    checkpoint that plays every field seat (net against net), and field_name is its label.
    field_name MIXED_FIELD with a field_checkpoint: the mixed table instead, that net in one
    field seat and MIXED_FIELD_AGENTS in the others (mixed_table_differences).
    Each net is built from its own run's config, so the two may differ in size and window.
    Returns a dict: checkpoint (file name), field, mean, ci_low, ci_high (95% bootstrap CI of
    the mean), standard_deviation, seconds (play time), differences (list of num_deals floats,
    the per-deal paired differences in deal order). Positive means the checkpoint is stronger.

    Every checkpoint sees the same games: the deals come from seed alone, and each rotation's
    rng is named by seed, deal and rotation, never by the checkpoint. So two checkpoints'
    differences can be compared deal by deal.
    """
    checkpoint_path, field_name, num_players, num_deals, seed, run_config, field_checkpoint = task

    # Many workers run at once, one per core. Torch's default would give each of them a thread
    # per core, all waiting on each other.
    torch.set_num_threads(1)

    net, history_window = load_checkpoint_net(checkpoint_path, run_config)
    test_agent = functools.partial(checkpoint_agent, net, history_window)
    field_agents = None
    if field_checkpoint is None:
        field_agent = AGENTS[field_name]
    else:
        field_path, field_config = field_checkpoint
        field_net, field_window = load_checkpoint_net(field_path, field_config)
        field_agent = functools.partial(checkpoint_agent, field_net, field_window)
        # The mixed table: the field net takes the last field seat, the agents the others.
        if field_name == MIXED_FIELD:
            field_agents = []
            for agent_name in MIXED_FIELD_AGENTS:
                field_agents.append(AGENTS[agent_name])
            field_agents.append(field_agent)

    # False: dev deals. It is fixed here, so no argument can reach the sealed set, and
    # select_deals raises on a seed below 10.
    deck_orders = select_deals(num_players, num_deals, seed, False)

    start = time.perf_counter()
    if field_agents is None:
        differences = paired_differences(
            deck_orders, num_players, test_agent, field_agent, TEST_SEATS, seed
        )
    else:
        differences = mixed_table_differences(
            deck_orders, num_players, test_agent, field_agents, seed
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
    parser.add_argument("--fields", help="field agents, comma-separated; default random,greedy")
    # Net against net: a second run's checkpoint plays every field seat. Both flags or neither.
    parser.add_argument("--field-run", help="run whose checkpoint is the field")
    parser.add_argument("--field-checkpoint", type=int, help="that checkpoint's updates done")
    # The mixed table: that net in one field seat, heuristic and greedy in the other two.
    parser.add_argument(
        "--mixed", action="store_true", help="mixed table with --field-run's net (stage 5)"
    )
    parser.add_argument("--deals", type=int, default=1000, help="deals per (checkpoint, field)")
    parser.add_argument("--seed", type=int, default=22, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--checkpoints", help="updates-done numbers, e.g. 0,3125; default all")
    parser.add_argument("--workers", type=int, help="processes; default the usable cores")
    args = parser.parse_args()

    # select_deals refuses the seed too, but inside a worker, after the pool has started.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(f"dev seeds are ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}")

    # A net field is its own evaluation: mixing it with field agents would put two kinds of
    # result in one file.
    if (args.field_run is None) != (args.field_checkpoint is None):
        raise SystemExit("--field-run and --field-checkpoint go together")
    if args.mixed and args.field_run is None:
        raise SystemExit("--mixed needs --field-run and --field-checkpoint: the table's net")
    field_checkpoint = None
    if args.field_run is not None:
        if args.fields is not None:
            raise SystemExit("--fields and --field-run are separate evaluations; give one")
        field_dir = RUNS_DIR / args.field_run
        field_path = field_dir / f"checkpoint_{args.field_checkpoint:06d}.pt"
        if not field_path.exists():
            raise SystemExit(f"{field_path} not found")
        with open(field_dir / "config.json") as file:
            field_config = json.load(file)
        field_checkpoint = (str(field_path), field_config)
        if args.mixed:
            field_names = [MIXED_FIELD]
        else:
            field_names = ["net"]
    elif args.fields is None:
        field_names = ["random", "greedy"]
    else:
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
            tasks.append(
                (
                    str(path),
                    field_name,
                    run_config["players"],
                    args.deals,
                    args.seed,
                    run_config,
                    field_checkpoint,
                )
            )

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
    # A net-against-net result gets its own name, so it cannot replace the curve against agents.
    if field_checkpoint is not None:
        if args.mixed:
            name += "_" + MIXED_FIELD
        name += f"_vs_{args.field_run}_ckpt{args.field_checkpoint}"
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
    if field_checkpoint is not None:
        summary["field_run"] = args.field_run
        summary["field_checkpoint"] = args.field_checkpoint
        summary["field_run_git_commit"] = field_checkpoint[1]["git_commit"]
    with open(RESULTS_DIR / (name + ".json"), "w") as file:
        json.dump(summary, file, indent=2)
    print(f"{name}: {len(tasks)} evaluations in {total_seconds:.1f} s on {workers} workers")


if __name__ == "__main__":
    main()
