"""
Same-curve check (stage plan specialist_explore, stage 5 piece 15): does a new run land inside a
reference run's 95% intervals at every checkpoint the two share?

Both curves are evaluate_checkpoints.py curve files: one row per (checkpoint, field), each with
the mean points per deal over the field and its 95% bootstrap interval over deals. For every
(checkpoint, field) in both files, the new run's mean must lie inside the reference's
[ci_low, ci_high], edges counted inside. One miss fails the check (Nate's rule, 2026-11-25): a
code change that bends the learning curve shows up as a mean outside the band at some point.

Run from the repo root, e.g. the same-curve run against stage 4's 10M smoke run:
    PYTHONPATH=src .venv/bin/python scripts/same_curve_check.py \
        --run 4p1d_dmc_teapot_a24_u3125_seed20_same-curve
Output: results/specialist_explore/<run>_same_curve_vs_<reference>.json (points matched, each
miss, the verdict, git commit). Exit 0 on a pass; exit 1 on any miss or no shared point.
"""

import argparse
import csv
import json
import sys

from evaluate_agents import RESULTS_ROOT, git_state

# Where evaluate_checkpoints.py writes curves, and the dev deal set both curves were scored on.
RESULTS_DIR = RESULTS_ROOT / "specialist_explore"
EVAL_DEALS = 1000
EVAL_SEED = 22

# Stage 4's 10M smoke run: seed 20, default arguments, a checkpoint every 125 updates.
REFERENCE_RUN = "4p1d_dmc_teapot_a24_u3125_seed20"

# Exit codes: the shell chain reads a fail as "stop".
EXIT_PASS = 0
EXIT_FAIL = 1


def points_outside(reference_rows, new_rows):
    """
    Compares a new curve with a reference curve at every (checkpoint, field) they share.
    reference_rows, new_rows: lists of dicts with int checkpoint and samples, str field, and
    float mean, ci_low, ci_high (the reference's interval is the one used).
    Returns (matched, misses): matched is the number of shared (checkpoint, field) points;
    misses is a list of dicts, one per new mean outside the reference interval, with the
    checkpoint, field, mean, interval and how far outside it lies.
    Raises ValueError when a shared checkpoint has different sample counts in the two runs:
    the same update number then means a different amount of training, so the points don't match.
    """
    # Index the reference by (checkpoint, field) so each new row finds its partner directly.
    reference_by_point = {}
    for row in reference_rows:
        reference_by_point[(row["checkpoint"], row["field"])] = row

    matched = 0
    misses = []
    for row in new_rows:
        point = (row["checkpoint"], row["field"])
        if point not in reference_by_point:
            continue
        reference = reference_by_point[point]
        if reference["samples"] != row["samples"]:
            raise ValueError(
                f"checkpoint {row['checkpoint']}: {reference['samples']} samples in the "
                f"reference, {row['samples']} in the new run"
            )
        matched += 1

        # Distance outside the band: 0 inside (edges included), positive past either edge.
        if row["mean"] < reference["ci_low"]:
            distance = reference["ci_low"] - row["mean"]
        elif row["mean"] > reference["ci_high"]:
            distance = row["mean"] - reference["ci_high"]
        else:
            distance = 0.0
        if distance > 0:
            misses.append({
                "checkpoint": row["checkpoint"],
                "field": row["field"],
                "mean": row["mean"],
                "ci_low": reference["ci_low"],
                "ci_high": reference["ci_high"],
                "distance": distance,
            })
    return matched, misses


def read_curve(run):
    """A run's curve file as a list of rows with typed values (see points_outside)."""
    path = RESULTS_DIR / f"{run}_eval_dev_n{EVAL_DEALS}_seed{EVAL_SEED}.csv"
    rows = []
    with open(path) as file:
        for raw in csv.DictReader(file):
            rows.append({
                "checkpoint": int(raw["checkpoint"]),
                "samples": int(raw["samples"]),
                "field": raw["field"],
                "mean": float(raw["mean"]),
                "ci_low": float(raw["ci_low"]),
                "ci_high": float(raw["ci_high"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="the new run, as named in its curve file")
    parser.add_argument("--reference", default=REFERENCE_RUN, help="the run whose intervals count")
    args = parser.parse_args()

    matched, misses = points_outside(read_curve(args.reference), read_curve(args.run))
    # No shared point means nothing was checked, which is a fail, not a pass.
    if matched > 0 and len(misses) == 0:
        verdict = "pass"
    else:
        verdict = "fail"

    commit, uncommitted_changes = git_state()
    record = {
        "run": args.run,
        "reference": args.reference,
        "matched": matched,
        "misses": misses,
        "verdict": verdict,
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
    }
    out_path = RESULTS_DIR / f"{args.run}_same_curve_vs_{args.reference}.json"
    with open(out_path, "w") as file:
        json.dump(record, file, indent=2)

    print(f"matched: {matched}")
    print(f"misses: {len(misses)}")
    for miss in misses:
        print(f"  checkpoint {miss['checkpoint']} vs {miss['field']}: mean {miss['mean']:.3f} "
              f"outside [{miss['ci_low']:.3f}, {miss['ci_high']:.3f}] by {miss['distance']:.3f}")
    print(f"verdict: {verdict}")
    print(f"wrote {out_path}")
    if verdict == "pass":
        sys.exit(EXIT_PASS)
    else:
        sys.exit(EXIT_FAIL)


if __name__ == "__main__":
    main()
