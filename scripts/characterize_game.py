"""
Game-shape measurement for one config and one agent (architecture plan §5 M2; stage plan
characterization, piece 2).

Plays --hands fresh dev hands with --agent in every seat and writes what the game looks like:
moves per hand, legal moves per turn, and pass rate (see stacker.characterize.game_shape).
Exploration only (rule 2): dev seeds ≥ 10, never sealed deals.

Run from the repo root, e.g. the heuristic at every seat of 4p1d, 1,000 hands:
    PYTHONPATH=src .venv/bin/python scripts/characterize_game.py --players 4 \
        --agent heuristic --hands 1000 --seed 13
Output: <out-dir>/<config>_game-shape_<agent>_dev_n<N>_seed<s>.csv (one row per hand) and .json
(arguments, summary, git commit, timings). The run name holds every argument, so runs never collide.
"""

import argparse
import csv
import json
import time
from pathlib import Path

# evaluate_agents.py sits beside this script, and Python puts the script's folder on the import
# path, so its agent table, seed floor and git check are reused rather than copied.
from evaluate_agents import AGENTS, DEV_SEED_MIN, RESULTS_ROOT, git_state

from stacker.characterize import game_shape
from stacker.sealed import config_name

# Agents measured in M2 (decision log 2026-11-07, "game-shape stats use the heuristic and
# greedy"). Random is left out: it makes hands unrealistically long.
SHAPE_AGENTS = ("greedy", "heuristic")

# Default output folder: the manifest's results/<slug>/ for this milestone.
DEFAULT_OUT_DIR = RESULTS_ROOT / "characterization"


def main():
    """Parses the command line, plays the hands, and writes the per-hand CSV and summary JSON."""
    parser = argparse.ArgumentParser(description="Game shape: one agent in every seat.")
    parser.add_argument("--players", type=int, required=True, help="3 to 8")
    parser.add_argument("--agent", choices=SHAPE_AGENTS, required=True)
    parser.add_argument("--hands", type=int, required=True, help="number of hands, at least 2")
    parser.add_argument("--seed", type=int, required=True, help=f"dev seed, ≥ {DEV_SEED_MIN}")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Rule 2: seeds 0-2 are test seeds, used only in confirm runs; M2 is exploration.
    if args.seed < DEV_SEED_MIN:
        raise SystemExit(
            f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {args.seed}"
        )

    run_name = (
        f"{config_name(args.players)}_game-shape_{args.agent}_dev_n{args.hands}_seed{args.seed}"
    )
    csv_path = args.out_dir / (run_name + ".csv")
    json_path = args.out_dir / (run_name + ".json")

    start = time.perf_counter()
    hand_rows, summary = game_shape(args.players, AGENTS[args.agent], args.hands, args.seed)
    play_seconds = time.perf_counter() - start

    commit, uncommitted_changes = git_state()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(hand_rows[0].keys()))
        writer.writeheader()
        for row in hand_rows:
            writer.writerow(row)

    # Seconds per hand is the probe number the budget rule needs for the full run's estimate.
    record = {
        "players": args.players,
        "agent": args.agent,
        "hands": args.hands,
        "seed": args.seed,
        "source": "dev",
        "summary": summary,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
        "play_seconds": round(play_seconds, 2),
        "seconds_per_hand": round(play_seconds / args.hands, 4),
    }
    with open(json_path, "w") as file:
        json.dump(record, file, indent=2)

    print(
        f"{run_name}: moves/hand {summary['moves_mean']:.1f} (p10 {summary['moves_p10']:.0f}, "
        f"p90 {summary['moves_p90']:.0f}), legal/turn {summary['legal_mean']:.1f} "
        f"(p90 {summary['legal_p90']:.0f}, max {summary['legal_max']}), "
        f"pass rate {summary['pass_rate']:.3f}, {play_seconds / args.hands:.3f} s/hand"
    )


if __name__ == "__main__":
    main()
