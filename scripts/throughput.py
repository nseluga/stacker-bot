"""
Simulator throughput: agent decisions per second on this machine (architecture plan §5 M2; stage
plan characterization, piece 5). The ceiling for M3's self-play actors before a net's cost.

Runs stacker.characterize.decisions_per_second once per process count in --processes and writes
one JSON for the sweep. Exploration only (rule 2): dev seeds ≥ 10.

Run from the repo root, e.g. 4p1d greedy at 1, 4 and 8 processes, 10 s each:
    PYTHONPATH=src .venv/bin/python scripts/throughput.py --players 4 --agent greedy \
        --seconds 10 --processes 1,4,8 --seed 14
Output: <out-dir>/<config>_throughput_<agent>_<machine>_dev_s<seconds>_seed<s>.json (machine,
CPU counts, Python version, one row per process count with load and wall time, git commit).
"""

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

from evaluate_agents import AGENTS, DEV_SEED_MIN, RESULTS_ROOT, git_state

from stacker.characterize import decisions_per_second
from stacker.sealed import config_name

# Greedy costs almost nothing per decision, so the timing is the engine's (stage plan piece 5).
# ponytail: greedy only; open the choice when the net exists and M3 measures actor cost.
THROUGHPUT_AGENTS = ("greedy",)

DEFAULT_OUT_DIR = RESULTS_ROOT / "characterization"


def main():
    """Parses the command line, runs one measurement per process count, writes the sweep JSON."""
    parser = argparse.ArgumentParser(description="Agent decisions per second on this machine.")
    parser.add_argument("--players", type=int, required=True, help="3 to 8")
    parser.add_argument("--agent", choices=THROUGHPUT_AGENTS, required=True)
    parser.add_argument("--seconds", type=float, required=True, help="per worker, per row")
    parser.add_argument("--processes", required=True, help="process counts, e.g. 1,4,8")
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Rule 2: seeds 0-2 are test seeds, used only in confirm runs; M2 is exploration.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )

    process_counts = []
    for part in args.processes.split(","):
        process_counts.append(int(part))

    # The short hostname names the machine, so laptop and teapot results never collide.
    # ponytail: macOS renames the laptop per network (…-133, …-89); add a --machine flag if a
    # stable laptop name matters.
    machine = socket.gethostname().split(".")[0]
    seconds_label = f"{args.seconds:g}"
    run_name = (
        f"{config_name(args.players)}_throughput_{args.agent}_{machine}"
        f"_dev_s{seconds_label}_seed{args.seed}"
    )

    rows = []
    for processes in process_counts:
        # Load before the row: other users' work on a shared box slows every worker.
        load_one_minute = os.getloadavg()[0]
        start = time.perf_counter()
        decision_rate, hand_rate, per_worker = decisions_per_second(
            args.players, AGENTS[args.agent], args.seconds, processes, args.seed
        )
        # Wall time includes pool start; it checks that workers' timed windows overlapped.
        wall_seconds = time.perf_counter() - start
        workers = []
        for decisions, hands, elapsed in per_worker:
            workers.append({"decisions": decisions, "hands": hands, "seconds": round(elapsed, 3)})
        rows.append(
            {
                "processes": processes,
                "load_one_minute_before": round(load_one_minute, 2),
                "wall_seconds": round(wall_seconds, 3),
                "decisions_per_second": round(decision_rate, 1),
                "hands_per_second": round(hand_rate, 2),
                "decisions_per_second_per_process": round(decision_rate / processes, 1),
                "workers": workers,
            }
        )
        # Per process shows how the rate scales: flat means each added process adds a full share.
        print(
            f"{run_name} p{processes}: {decision_rate:,.0f} decisions/s, {hand_rate:,.1f} hands/s, "
            f"{decision_rate / processes:,.0f} per process"
        )

    commit, uncommitted_changes = git_state()

    # CPUs this process may run on: an account limit can be below cpu_count (teapot: 39 of 48).
    # sched_getaffinity is Linux only, so macOS records None.
    # ponytail: physical-core layout (lscpu) is cited in the log, not recorded here.
    if hasattr(os, "sched_getaffinity"):
        usable_cpus = len(os.sched_getaffinity(0))
    else:
        usable_cpus = None

    record = {
        "players": args.players,
        "agent": args.agent,
        "seconds": args.seconds,
        "seed": args.seed,
        "source": "dev",
        "machine": machine,
        "cpu_count": os.cpu_count(),
        "usable_cpus": usable_cpus,
        "python": platform.python_version(),
        "rows": rows,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / (run_name + ".json"), "w") as file:
        json.dump(record, file, indent=2)


if __name__ == "__main__":
    main()
