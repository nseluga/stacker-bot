"""
Deals needed against the edge to detect (architecture plan §5 M2; stage plan characterization,
piece 4). Reads σ from a deal-luck run's JSON and writes N = ceil((2.8σ/δ)²) for each δ.

Run from the repo root after scripts/deal_luck.py:
    PYTHONPATH=src .venv/bin/python scripts/sigma_to_n.py \
        results/characterization/4p1d_deal-luck_heuristic-vs-greedy_dev_n1000_seed12.json
Output: results/characterization/<config>_sigma-to-n.csv (delta, sigma, deals_needed).
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate_agents import RESULTS_ROOT

from stacker.characterize import deals_needed
from stacker.sealed import config_name

# The edges in the decision log entry "M2 measures σ → N": 0.10 is M1's δ, the rest bracket it.
DELTAS = (0.03, 0.05, 0.10, 0.20)


def main():
    """Reads σ from the deal-luck JSON and writes one CSV row per δ."""
    parser = argparse.ArgumentParser(description="Deals needed for each δ, from a measured σ.")
    parser.add_argument("deal_luck_json", type=Path, help="a scripts/deal_luck.py JSON")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "characterization")
    args = parser.parse_args()

    with open(args.deal_luck_json) as file:
        record = json.load(file)
    sigma = record["standard_deviation"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{config_name(record['players'])}_sigma-to-n.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delta", "sigma", "deals_needed", "source"])
        for delta in DELTAS:
            needed = deals_needed(sigma, delta)
            writer.writerow([delta, sigma, needed, args.deal_luck_json.name])
            print(f"δ {delta:.2f}: N = {needed}")


if __name__ == "__main__":
    main()
