"""
Plots one training run's learning curve from the file scripts/evaluate_checkpoints.py writes
(stage plan specialist_explore, stage 4 piece 14).

One panel: training samples seen (x) against the checkpoint's mean points per deal over the
field (y), one line per field agent, the 95% bootstrap CI as a band, and 0 marked. A band clear
of 0 is an edge on these dev deals (exploration, not evidence).

It also prints, per point, the standard deviation of the per-deal differences and the smallest
edge that many deals can detect, 2.8 σ / √N (2.8 = 1.96 + 0.84: a two-sided 5% false alarm rate
and 80% power; stage plan baselines, Colas 2018). The deal count was planned from σ = 0.376.

Run from the repo root:
    .venv/bin/python scripts/plot_checkpoint_curve.py --curve results/specialist_explore/<name>.csv
Reads <name>.csv and <name>.json (for the deal count). Writes <name>.png beside them.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Power-analysis multiplier: z for a two-sided 5% test (1.96) plus z for 80% power (0.84).
POWER_MULTIPLIER = 2.8
# The x-axis is in millions of samples.
SAMPLES_PER_MILLION = 1_000_000

# Colors: categorical slots of the dataviz reference palette, as in plot_pilot.py; a field not
# listed here gets matplotlib's next default color. Text and grid in neutrals.
FIELD_COLORS = {"random": "#2a78d6", "greedy": "#eb6834"}
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
BAND_ALPHA = 0.2


def plot_curve(curve_path):
    """
    Draws the curve file's points and prints each point's σ and detectable edge.
    curve_path: Path of a <name>.csv from evaluate_checkpoints.py (columns checkpoint, samples,
    field, mean, ci_low, ci_high, standard_deviation, seconds), with <name>.json beside it.
    Returns the Path of the PNG it wrote.
    """
    summary = json.loads(curve_path.with_suffix(".json").read_text())
    deals = summary["deals"]

    # rows_by_field[field] = that field's rows in file order, which is training order.
    rows_by_field = {}
    with open(curve_path) as handle:
        for row in csv.DictReader(handle):
            rows_by_field.setdefault(row["field"], []).append(row)

    figure, axis = plt.subplots(figsize=(9, 5))
    for field, rows in rows_by_field.items():
        millions = []
        means = []
        lows = []
        highs = []
        for row in rows:
            millions.append(int(row["samples"]) / SAMPLES_PER_MILLION)
            means.append(float(row["mean"]))
            lows.append(float(row["ci_low"]))
            highs.append(float(row["ci_high"]))
            sigma = float(row["standard_deviation"])
            detectable = POWER_MULTIPLIER * sigma / math.sqrt(deals)
            print(
                f"{int(row['samples']):>12,} samples vs {field}: mean {float(row['mean']):+.3f}, "
                f"σ {sigma:.3f}, detects {detectable:.3f}"
            )
        # None: matplotlib picks its next default color.
        color = FIELD_COLORS.get(field)
        (line,) = axis.plot(
            millions, means, marker="o", markersize=4, linewidth=2, color=color,
            label=f"vs {field}",
        )
        # The band takes the line's color, whichever way it was chosen.
        axis.fill_between(millions, lows, highs, color=line.get_color(), alpha=BAND_ALPHA,
                          linewidth=0)

    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axis.set_xlabel("training samples seen (millions)")
    axis.set_ylabel("mean points per deal over the field (95% bootstrap CI)")
    axis.set_title(
        f"{summary['run']}: {deals} dev deals, seed {summary['seed']}", loc="left", fontsize=11
    )
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.legend(frameon=False, fontsize=9, loc="lower right")

    figure.tight_layout()
    plot_path = curve_path.with_suffix(".png")
    figure.savefig(plot_path, dpi=150)
    return plot_path


def main():
    """Parses the command line and plots the named curve file."""
    parser = argparse.ArgumentParser(description="Plot a run's checkpoint curve.")
    parser.add_argument("--curve", required=True, type=Path, help="the curve .csv")
    args = parser.parse_args()
    print(f"wrote {plot_curve(args.curve)}")


if __name__ == "__main__":
    main()
