"""
Plots the baselines probe and pilot runs, to read before δ and N are chosen (stage plan baselines,
pieces 9-10).

Three panels, one question each:
1. What does one deal's result look like? Share of deals at each per-deal difference (the test
   seat's points minus the field's share, averaged over the 4 seat rotations), for the pilot
   (heuristic vs greedy) and the sanity rung (greedy vs random).
2. How sure are we of each mean? Mean and percentile-bootstrap 95% CI of every run, against 0.
3. What does δ cost? N = (2.8 σ / δ)² from the pilot σ: 2.8 = 1.96 + 0.84, a two-sided 5% false
   alarm rate and 80% power at an edge of δ (stage plan baselines, Colas 2018). Each marked δ
   also shows the confirm's play time: N deals x 3 seeds at the measured seconds per deal.

Run from the repo root:  PYTHONPATH=src .venv/bin/python scripts/plot_pilot.py
Output: results/baselines/pilot.png
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results" / "baselines"
PILOT = "4p1d_heuristic-vs-greedy_seats0_dev_n1000_seed11"
PROBE = "4p1d_heuristic-vs-greedy_seats0_dev_n400_seed10"
SANITY = "4p1d_greedy-vs-random_seats0_dev_n400_seed10"

# Power-analysis multiplier: z for a two-sided 5% test (1.96) plus z for 80% power (0.84).
POWER_MULTIPLIER = 2.8
# Confirm runs at test seeds 0, 1 and 2 (rule 4: at least 3 seeds).
CONFIRM_SEEDS = 3
# The δ values discussed so far, marked on the cost curve.
MARKED_DELTAS = (0.2, 0.1, 0.05)

# Colors: categorical slots 1 and 2 of the dataviz reference palette; text and grid in neutrals.
HEURISTIC_COLOR = "#2a78d6"
GREEDY_COLOR = "#eb6834"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def load_run(name):
    """A run's summary (the .json) and its per-deal differences (the .csv), by run name."""
    summary = json.loads((RESULTS / f"{name}.json").read_text())
    differences = []
    with open(RESULTS / f"{name}.csv") as handle:
        for row in csv.DictReader(handle):
            differences.append(float(row["difference"]))
    return summary, differences


def shares_by_value(differences):
    """Share of deals at each distinct difference value; returns (values, shares), values sorted."""
    counts = {}
    for value in differences:
        counts[value] = counts.get(value, 0) + 1
    values = sorted(counts)
    shares = []
    for value in values:
        shares.append(counts[value] / len(differences))
    return values, shares


def style(axis, title):
    """Recessive grid and spines, left-aligned title: the data marks carry the chart."""
    axis.set_title(title, loc="left", fontsize=11)
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)


pilot, pilot_differences = load_run(PILOT)
probe, _probe_differences = load_run(PROBE)
sanity, sanity_differences = load_run(SANITY)
sigma = pilot["standard_deviation"]
seconds_per_deal = pilot["play_seconds"] / pilot["deals"]

figure, (panel_spread, panel_means, panel_cost) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: the per-deal distributions, side by side at each value (shares, so n=1000 and n=400
# compare directly). Differences move in steps of 0.25: one rotation's point, split 4 ways.
bar_width = 0.1
for offset, differences, color, label in (
    (-bar_width / 2, pilot_differences, HEURISTIC_COLOR, "heuristic vs greedy (pilot, n=1000)"),
    (bar_width / 2, sanity_differences, GREEDY_COLOR, "greedy vs random (sanity, n=400)"),
):
    values, shares = shares_by_value(differences)
    positions = []
    for value in values:
        positions.append(value + offset)
    panel_spread.bar(positions, shares, width=bar_width * 0.9, color=color, label=label)
panel_spread.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
panel_spread.set_xlabel("per-deal difference (points per hand, test − field share)")
panel_spread.set_ylabel("share of deals")
panel_spread.legend(frameon=False, fontsize=9, loc="upper left")
style(panel_spread, "1. One deal's result: right of 0 = the test agent did better")

# Panel 2: each run's mean with its 95% CI as a horizontal bar, 0 marked: a CI clear of 0 is a
# real edge on these dev deals (exploration, not evidence).
runs = (
    ("greedy vs random\nsanity, seed 10, n=400", sanity, GREEDY_COLOR),
    ("heuristic vs greedy\nprobe, seed 10, n=400", probe, HEURISTIC_COLOR),
    ("heuristic vs greedy\npilot, seed 11, n=1000", pilot, HEURISTIC_COLOR),
)
for row, (label, summary, color) in enumerate(runs):
    low_error = summary["mean"] - summary["ci_low"]
    high_error = summary["ci_high"] - summary["mean"]
    panel_means.errorbar(
        summary["mean"], row, xerr=[[low_error], [high_error]], fmt="o", color=color,
        markersize=8, capsize=5, linewidth=2,
    )
    text = f"{summary['mean']:+.3f}  [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]"
    panel_means.annotate(text, (summary["ci_high"], row), xytext=(8, 0),
                         textcoords="offset points", va="center", fontsize=9)
panel_means.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
run_labels = []
for label, _summary, _color in runs:
    run_labels.append(label)
panel_means.set_yticks(range(len(runs)))
panel_means.set_yticklabels(run_labels, fontsize=9)
panel_means.set_xlim(-0.05, 1.0)
panel_means.set_ylim(-0.6, len(runs) - 0.4)
panel_means.set_xlabel("mean points per hand over the field (95% bootstrap CI)")
style(panel_means, "2. Mean edge per run, with 95% CI")

# Panel 3: N against δ on a log scale (N grows as 1/δ²), from the pilot σ, with the marked δs.
deltas = []
sizes = []
step = 0.02
while step <= 0.3001:
    deltas.append(step)
    sizes.append((POWER_MULTIPLIER * sigma / step) ** 2)
    step += 0.005
panel_cost.plot(deltas, sizes, color=HEURISTIC_COLOR, linewidth=2)
for delta in MARKED_DELTAS:
    deals = (POWER_MULTIPLIER * sigma / delta) ** 2
    minutes = deals * CONFIRM_SEEDS * seconds_per_deal / 60
    panel_cost.plot(delta, deals, "o", color=HEURISTIC_COLOR, markersize=8)
    panel_cost.annotate(f"δ={delta}: N≈{deals:.0f}, confirm ≈{minutes:.1f} min",
                        (delta, deals), xytext=(8, 6), textcoords="offset points", fontsize=9)
panel_cost.set_yscale("log")
panel_cost.set_xlabel("δ, smallest edge to detect (points per hand)")
panel_cost.set_ylabel("deals needed, N (log scale)")
style(panel_cost, f"3. Deals needed vs δ (pilot σ = {sigma:.3f}, 80% power)")

figure.tight_layout()
figure.savefig(RESULTS / "pilot.png", dpi=150)
print(f"wrote {RESULTS / 'pilot.png'}  (σ={sigma:.4f}, {seconds_per_deal:.3f} s/deal)")
