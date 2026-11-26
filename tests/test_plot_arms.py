"""
Gate for scripts/plot_arms.py (stage plan specialist_explore, stage 5 piece 8): the labels and
six plots from a synthetic arms list, curves, deal files and logs.
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import plot_arms
from test_net import DEV_SEED

NUM_DEALS = 50
# Checkpoints every 10 updates; the arm length is 20, so 0, 10, 20 count.
CHECKPOINT_EVERY = 10
ARM_LENGTH = 20
SAMPLES_PER_UPDATE = 3200
# Control seed 20 is the long run: it carries a checkpoint past arm length, planted huge, so a
# script that reads it at the end of its file gets a different label.
PLANTED_LATE_SCORE = 5.0
# Per-deal spread shared by every run (deal luck), and each run's own small noise.
DEAL_SPREAD = 0.3
RUN_NOISE = 0.01
# Half-width of the synthetic curve intervals; only drawn, never tested.
CURVE_HALF_WIDTH = 0.05
# Control seeds at 0, 0.1, 0.2; the wide arm at 0.3 (both seeds), the narrow arm at -0.1.
CONTROL_OFFSETS = (0.0, 0.1, 0.2)
WIDE_OFFSET = 0.3
NARROW_OFFSET = -0.1
# The plots the decision log names.
NUM_PLOTS = 6


def write_run(results_dir, runs_dir, run, offset, last_checkpoint, late_score=None):
    """One synthetic run: curve, per-deal file and log, every deal at base noise + offset."""
    rng = random.Random(DEV_SEED)
    base = []
    for _ in range(NUM_DEALS):
        base.append(rng.gauss(0.0, DEAL_SPREAD))
    noise = random.Random(run)
    stem = f"{run}_eval_dev_n{plot_arms.EVAL_DEALS}_seed{plot_arms.EVAL_SEED}"
    with open(results_dir / f"{stem}.csv", "w", newline="") as curve_file, \
            open(results_dir / f"{stem}_deals.csv", "w", newline="") as deals_file:
        curve = csv.writer(curve_file)
        deals = csv.writer(deals_file)
        curve.writerow(["checkpoint", "samples", "field", "mean", "ci_low", "ci_high"])
        deals.writerow(["checkpoint", "field", "deal_index", "difference"])
        for checkpoint in range(0, last_checkpoint + 1, CHECKPOINT_EVERY):
            for field in plot_arms.FIELDS:
                values = []
                for deal_index in range(NUM_DEALS):
                    value = base[deal_index] + offset + noise.gauss(0.0, RUN_NOISE)
                    if late_score is not None and checkpoint > ARM_LENGTH:
                        value = late_score
                    values.append(value)
                    deals.writerow([checkpoint, field, deal_index, value])
                mean = sum(values) / NUM_DEALS
                curve.writerow([checkpoint, checkpoint * SAMPLES_PER_UPDATE, field, mean,
                                mean - CURVE_HALF_WIDTH, mean + CURVE_HALF_WIDTH])
    (runs_dir / run).mkdir(parents=True)
    with open(runs_dir / run / "log.jsonl", "w") as log:
        for update in range(1, last_checkpoint + 1):
            # Loss falls as 1 / update; every update takes 2 seconds.
            entry = {"update": update, "loss": 1.0 / update, "gradient_norm": 1.0,
                     "wait_seconds": 1.0, "learner_seconds": 1.0}
            log.write(json.dumps(entry) + "\n")


def test_labels_read_the_control_at_arm_length_and_all_six_plots_are_written(tmp_path):
    # catches: the long control seed read at the end of its file instead of at arm length
    # (its late checkpoint is planted huge, which turns the candidate into a tie);
    # catches: a plot silently skipped.
    results_dir = tmp_path / "results"
    runs_dir = tmp_path / "runs"
    results_dir.mkdir()
    long_control_length = 2 * ARM_LENGTH
    write_run(results_dir, runs_dir, "control-20", CONTROL_OFFSETS[0], long_control_length,
              PLANTED_LATE_SCORE)
    write_run(results_dir, runs_dir, "control-21", CONTROL_OFFSETS[1], ARM_LENGTH)
    write_run(results_dir, runs_dir, "control-23", CONTROL_OFFSETS[2], ARM_LENGTH)
    write_run(results_dir, runs_dir, "stage4-60M", CONTROL_OFFSETS[0], ARM_LENGTH)
    write_run(results_dir, runs_dir, "wide-20", WIDE_OFFSET, ARM_LENGTH)
    write_run(results_dir, runs_dir, "wide-21", WIDE_OFFSET, ARM_LENGTH)
    write_run(results_dir, runs_dir, "narrow-20", NARROW_OFFSET, ARM_LENGTH)
    arms = {
        "arm_length_updates": ARM_LENGTH,
        "control": ["control-20", "control-21", "control-23"],
        "repeat": {"long": "control-20", "short": "stage4-60M"},
        "families": {"width": [
            {"label": "W1024", "run": "wide-20", "second_seed_run": "wide-21"},
            {"label": "W256", "run": "narrow-20", "second_seed_run": None},
        ]},
    }
    arms_json = tmp_path / "arms.json"
    arms_json.write_text(json.dumps(arms))

    written = plot_arms.make_all(arms_json, results_dir, runs_dir)

    labels = {}
    with open(results_dir / "arm_verdicts.csv") as file:
        for row in csv.DictReader(file):
            labels[(row["label"], row["field"])] = row["verdict"]
    assert labels == {
        ("W1024", "heuristic"): plot_arms.WINNER, ("W1024", "greedy"): plot_arms.WINNER,
        ("W256", "heuristic"): "loses", ("W256", "greedy"): "loses",
    }
    pngs = []
    for path in written:
        if path.suffix == ".png":
            pngs.append(path)
    assert len(pngs) == NUM_PLOTS
    for path in pngs:
        assert path.stat().st_size > 0


def write_mixed(results_dir, run, table, offset):
    """One synthetic per-deal file for one mixed table at checkpoints 0..ARM_LENGTH."""
    checkpoints = []
    for checkpoint in range(0, ARM_LENGTH + 1, CHECKPOINT_EVERY):
        checkpoints.append(checkpoint)
    path = plot_arms.mixed_deals_path(results_dir, run, table, checkpoints)
    noise = random.Random(f"{run}:{table}")
    with open(path, "w", newline="") as file:
        deals = csv.writer(file)
        deals.writerow(["checkpoint", "field", "deal_index", "difference"])
        for checkpoint in checkpoints:
            for deal_index in range(NUM_DEALS):
                value = offset + noise.gauss(0.0, RUN_NOISE)
                deals.writerow([checkpoint, table, deal_index, value])


def test_mixed_file_name_matches_what_evaluate_checkpoints_writes():
    # catches: plot_arms looking for a name the eval script never writes, so every mixed table
    # is silently skipped. The expected names are evaluate_checkpoints.py --mixed-table's format.
    styles = plot_arms.mixed_deals_path(Path("r"), "run", "mixed-styles", [35000, 36250, 37500])
    weak = plot_arms.mixed_deals_path(Path("r"), "run", "mixed-weak", [35000, 36250, 37500])
    assert styles.name == (
        "run_eval_dev_n1000_seed22_mixed-styles_vs_4p1d_dmc_teapot_a24_u18750_seed20_ckpt18750"
        "_ckpt35000-36250-37500_deals.csv"
    )
    assert weak.name == "run_eval_dev_n1000_seed22_mixed-weak_ckpt35000-36250-37500_deals.csv"


def test_each_mixed_table_is_labelled_only_once_every_control_played_it(tmp_path):
    # catches: a table labelled while a control seed has not played it (the control mean would
    # come from 2 seeds or crash); an arm or second seed without a table's file crashing the
    # script, or the arm called a winner on a second seed that never played the table.
    results_dir = tmp_path / "results"
    runs_dir = tmp_path / "runs"
    results_dir.mkdir()
    write_run(results_dir, runs_dir, "control-20", CONTROL_OFFSETS[0], 2 * ARM_LENGTH)
    write_run(results_dir, runs_dir, "control-21", CONTROL_OFFSETS[1], ARM_LENGTH)
    write_run(results_dir, runs_dir, "control-23", CONTROL_OFFSETS[2], ARM_LENGTH)
    write_run(results_dir, runs_dir, "stage4-60M", CONTROL_OFFSETS[0], ARM_LENGTH)
    write_run(results_dir, runs_dir, "wide-20", WIDE_OFFSET, ARM_LENGTH)
    write_run(results_dir, runs_dir, "wide-21", WIDE_OFFSET, ARM_LENGTH)
    write_run(results_dir, runs_dir, "narrow-20", NARROW_OFFSET, ARM_LENGTH)
    # Styles: all controls and the wide arm's first seed; wide-21 and narrow have not played it.
    # Weak: control seed 23 has not played it yet, so it must not be labelled for anyone.
    played = (
        ("mixed-styles", "control-20", CONTROL_OFFSETS[0]),
        ("mixed-styles", "control-21", CONTROL_OFFSETS[1]),
        ("mixed-styles", "control-23", CONTROL_OFFSETS[2]),
        ("mixed-styles", "wide-20", WIDE_OFFSET),
        ("mixed-weak", "control-20", CONTROL_OFFSETS[0]),
        ("mixed-weak", "control-21", CONTROL_OFFSETS[1]),
        ("mixed-weak", "wide-20", WIDE_OFFSET),
    )
    for table, run, offset in played:
        write_mixed(results_dir, run, table, offset)
    arms = {
        "arm_length_updates": ARM_LENGTH,
        "control": ["control-20", "control-21", "control-23"],
        "repeat": {"long": "control-20", "short": "stage4-60M"},
        "families": {"width": [
            {"label": "W1024", "run": "wide-20", "second_seed_run": "wide-21"},
            {"label": "W256", "run": "narrow-20", "second_seed_run": None},
        ]},
    }
    arms_json = tmp_path / "arms.json"
    arms_json.write_text(json.dumps(arms))

    written = plot_arms.make_all(arms_json, results_dir, runs_dir)

    labels_by_table = {}
    with open(results_dir / "arm_verdicts.csv") as file:
        for row in csv.DictReader(file):
            if row["field"] in plot_arms.MIXED_TABLES:
                if row["field"] not in labels_by_table:
                    labels_by_table[row["field"]] = {}
                labels_by_table[row["field"]][row["label"]] = row["verdict"]
    assert labels_by_table == {"mixed-styles": {"W1024": "candidate"}}
    assert results_dir / "arms_7_mixed_styles_end_scores.png" in written
    assert results_dir / "arms_8_mixed_weak_end_scores.png" not in written
