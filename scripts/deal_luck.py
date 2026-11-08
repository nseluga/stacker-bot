"""
Deal-luck share and σ of the 1v3 paired difference (architecture plan §5 M2; stage plan
characterization, piece 3).

Plays --deals dev deals once per seat rotation, the test agent in one seat and the field agent in
the rest, and keeps the test agent's points in every rotation. From those it reports how much of
one hand's spread duplicate pairing removes (stacker.characterize.deal_luck_share) and σ, the
input to the deals-needed table. Exploration only (rule 2): dev seeds ≥ 10, never sealed deals.

Run from the repo root, e.g. heuristic vs greedy, 4p1d, 1,000 deals:
    PYTHONPATH=src .venv/bin/python scripts/deal_luck.py --players 4 --test heuristic \
        --field greedy --deals 1000 --seed 12
Output: <out-dir>/<config>_deal-luck_<test>-vs-<field>_dev_n<N>_seed<s>.csv (deal_index and one
column per rotation) and .json (arguments, share, σ, mean difference, git commit, timings).
"""

import argparse
import csv
import json
import time
from pathlib import Path

# evaluate_agents.py sits beside this script, so its deal builder, agents and git check are reused.
from evaluate_agents import AGENTS, RESULTS_ROOT, git_state, select_deals

from stacker.characterize import deal_luck_share
from stacker.evaluate import POINTS_PER_HAND, rotation_scores
from stacker.sealed import config_name

# 1v3: the test agent holds seat 0 at rotation 0 (decision log 2026-11-07, "σ → N for 1v3 only").
# ponytail: format fixed to one test seat; add a --test-seats flag if 2v2 or 3v1 needs a share.
TEST_SEATS = (0,)

DEFAULT_OUT_DIR = RESULTS_ROOT / "characterization"


def main():
    """Parses the command line, plays every rotation, writes the per-deal CSV and summary JSON."""
    parser = argparse.ArgumentParser(description="Deal-luck share and σ, 1v3 test vs field.")
    parser.add_argument("--players", type=int, required=True, help="3 to 8")
    parser.add_argument("--test", choices=sorted(AGENTS), required=True)
    parser.add_argument("--field", choices=sorted(AGENTS), required=True)
    parser.add_argument("--deals", type=int, required=True, help="number of deals, at least 2")
    parser.add_argument("--seed", type=int, required=True, help="dev seed, ≥ 10")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # select_deals raises on a test seed (rule 2); turn that into a clean exit before any play.
    try:
        deck_orders = select_deals(args.players, args.deals, args.seed, False)
    except ValueError as error:
        raise SystemExit(str(error))

    run_name = (
        f"{config_name(args.players)}_deal-luck_{args.test}-vs-{args.field}"
        f"_dev_n{args.deals}_seed{args.seed}"
    )
    csv_path = args.out_dir / (run_name + ".csv")
    json_path = args.out_dir / (run_name + ".json")

    # Same rng naming as paired_differences, so deal i here is played exactly as an M1 run would.
    start = time.perf_counter()
    rotation_rows = []
    for deal_index, deck_order in enumerate(deck_orders):
        scores = rotation_scores(
            deck_order, args.players, AGENTS[args.test], AGENTS[args.field], TEST_SEATS,
            args.seed, deal_index,
        )
        rotation_rows.append(scores)
    play_seconds = time.perf_counter() - start

    share, sigma, seat_variance, deal_variance = deal_luck_share(rotation_rows)

    # Mean paired difference, as in paired_differences: rotation mean minus the field's share.
    field_share = len(TEST_SEATS) * POINTS_PER_HAND / args.players
    total = 0
    for scores in rotation_rows:
        total += sum(scores) / args.players - field_share
    mean_difference = total / len(rotation_rows)

    commit, uncommitted_changes = git_state()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["deal_index"]
        for rotation in range(args.players):
            header.append(f"rotation_{rotation}")
        writer.writerow(header)
        for deal_index, scores in enumerate(rotation_rows):
            writer.writerow([deal_index] + list(scores))

    record = {
        "players": args.players,
        "test": args.test,
        "field": args.field,
        "test_seats": list(TEST_SEATS),
        "deals": args.deals,
        "seed": args.seed,
        "source": "dev",
        "deal_luck_share": share,
        "standard_deviation": sigma,
        "seat_variance": seat_variance,
        "deal_variance": deal_variance,
        "mean_difference": mean_difference,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
        "play_seconds": round(play_seconds, 2),
        "seconds_per_deal": round(play_seconds / args.deals, 4),
    }
    with open(json_path, "w") as file:
        json.dump(record, file, indent=2)

    print(
        f"{run_name}: share {share}, sd {sigma:.4f}, mean diff {mean_difference:+.4f}, "
        f"{play_seconds / args.deals:.3f} s/deal"
    )


if __name__ == "__main__":
    main()
