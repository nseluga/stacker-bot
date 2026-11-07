"""
Summarizes the M1 confirm across its three test seeds and applies the pre-registered pass rule
(decision log 2026-11-07, "Pre-registration: M1 confirm, heuristic vs greedy 1v3 on 4p1d").

Each seed's run (scripts/evaluate_agents.py --confirm) already holds its own mean and 95% CI.
This script adds what rule 4 asks for across seeds: the mean ± sd of the three seed means, never
the best seed alone. Pass rule: every seed's CI lower bound is above 0; one seed at or below 0
fails the confirm.

Run from the repo root, after all three confirm runs:  .venv/bin/python scripts/summarize_confirm.py
Output: results/baselines/4p1d_heuristic-vs-greedy_seats0_sealed_n150_summary.csv, and the
verdict printed.
"""

import csv
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "baselines"
# The pre-registered runs: names as scripts/evaluate_agents.py writes them, one per test seed.
RUN_PREFIX = "4p1d_heuristic-vs-greedy_seats0_sealed_n150"
TEST_SEEDS = (0, 1, 2)


def summarize(run_names, out_path):
    """
    Reads each run's .json summary, writes one row per run plus a mean ± sd row to out_path,
    and returns (passed, rows). passed: True only when every run's ci_low is above 0.
    """
    rows = []
    for name in run_names:
        summary = json.loads((RESULTS / f"{name}.json").read_text())
        rows.append({
            "run": name,
            "seed": summary["seed"],
            "deals": summary["deals"],
            "mean": summary["mean"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "standard_deviation": summary["standard_deviation"],
            "uncommitted_changes": summary["uncommitted_changes"],
        })

    # The pass rule: a single seed whose CI reaches 0 fails the whole confirm.
    passed = True
    for row in rows:
        if row["ci_low"] <= 0:
            passed = False

    # Rule 4: the spread across seeds is reported beside their mean. Its own column, because
    # standard_deviation in the seed rows is the per-deal σ, a different quantity.
    means = []
    for row in rows:
        means.append(row["mean"])
    across = {"run": "mean ± sd over seeds", "mean": statistics.mean(means)}
    if len(means) > 1:
        across["sd_of_seed_means"] = statistics.stdev(means)

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["sd_of_seed_means"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(across)
    return passed, rows + [across]


if __name__ == "__main__":
    names = []
    for seed in TEST_SEEDS:
        names.append(f"{RUN_PREFIX}_seed{seed}")
    out_path = RESULTS / f"{RUN_PREFIX}_summary.csv"
    passed, rows = summarize(names, out_path)
    for row in rows:
        print(row)
    if passed:
        print(f"PASS: every seed's CI lower bound > 0. Wrote {out_path}")
    else:
        print(f"FAIL: a seed's CI lower bound ≤ 0. Wrote {out_path}")
