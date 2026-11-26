"""
Stage 5 (exploration arms) plots and labels, from result files and run logs only (stage plan
specialist_explore, stage 5 piece 8; decision log 2026-11-22, "what gets plotted" and "the rule
by which a result counts").

Input: a hand-edited arms list, results/specialist_explore/arms.json, updated as each run lands:
    {
      "arm_length_updates": 37500,               # Nate's call from the 300M curve; null before
      "control": ["<seed-20 run>", "<seed-21 run>", "<seed-23 run>"],
      "repeat": {"long": "<300M seed-20 run>", "short": "<stage 4 60M seed-20 run>"},
      "families": {
        "width": [{"label": "W1024", "run": "<seed-20 run>", "second_seed_run": null}, ...],
        ...
      }
    }
Per run it reads results/specialist_explore/<run>_eval_dev_n1000_seed22.csv and _deals.csv
(evaluate_checkpoints.py) and runs/specialist_explore/<run>/log.jsonl (train_dmc.py). runs/ is
gitignored and lives on teapot: run this there, or after an rsync.

Writes to results/specialist_explore/:
- arm_verdicts.csv: every arm's label per field (candidate / loses / tie, and winner when the
  second seed is also a candidate), with the interval. The labels are exploration only.
- arms_1_heuristic_by_samples.png, arms_2_greedy_by_samples.png, arms_3_heuristic_by_hours.png,
  arms_4_training.png, arms_5_end_scores.png, arms_6_repeat.png.
Before arm_length_updates is set, only plot 6 is drawn.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/plot_arms.py
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_checkpoints import RESULTS_DIR, RUNS_DIR
from stacker.arms import CANDIDATE, arm_verdict, end_scores

# The dev-deal evaluation every stage 5 run gets: 1,000 deals, seed 22.
EVAL_DEALS = 1000
EVAL_SEED = 22
# The fields scored; the rule's metric is the heuristic, greedy is shown beside it.
FIELDS = ("heuristic", "greedy")
# The mixed tables (evaluate_checkpoints.py --mixed-table; decision log 2026-11-26), by the field
# label their files carry. Reported beside the label, never in it. Each gives:
#   net_suffix: the file-name part naming the table's net seat ("" for a table with none),
#   title: the plot's title, plot: the plot's file name.
STYLES_NET_RUN = "4p1d_dmc_teapot_a24_u18750_seed20"
STYLES_NET_CHECKPOINT = 18750
MIXED_TABLES = {
    "mixed-styles": {
        "net_suffix": f"_vs_{STYLES_NET_RUN}_ckpt{STYLES_NET_CHECKPOINT}",
        "title": "styles table: heuristic, greedy, 60M net (beside the label, not in it)",
        "plot": "arms_7_mixed_styles_end_scores.png",
    },
    "mixed-weak": {
        "net_suffix": "",
        "title": "weak table: random, greedy, heuristic (beside the label, not in it)",
        "plot": "arms_8_mixed_weak_end_scores.png",
    },
}
# A run's score is the mean over its last 3 checkpoints at or before arm length.
END_CHECKPOINTS = 3
# The label for a candidate whose second seed is also a candidate.
WINNER = "winner"
# Plot 4 averages the per-update log over blocks of this many updates (raw is too noisy to read).
LOG_BLOCK = 100
SAMPLES_PER_MILLION = 1_000_000
SECONDS_PER_HOUR = 3600

# Colors: neutrals for the control band and text; arms take matplotlib's default cycle.
CONTROL_COLOR = "#8a8984"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
BAND_ALPHA = 0.25

# Figure sizes in inches, and the resolution the PNGs are saved at.
PANEL_WIDTH = 5
PANEL_HEIGHT = 4
TRAINING_FIGURE_SIZE = (11, 4)
REPEAT_FIGURE_SIZE = (9, 5)
END_SCORES_WIDTH = 8
# Plot 5 grows by this much height per arm, on top of a fixed base for the axis and title.
END_SCORES_ROW_HEIGHT = 0.4
END_SCORES_BASE_HEIGHT = 1.5
# Plot 5 leaves this share of the x range free on each side, for the rightmost arm's label.
END_SCORES_X_MARGIN = 0.15
# Plot 5 writes each label this many points to the right of the arm's interval.
LABEL_OFFSET_POINTS = 6
SAVE_DPI = 150

# Line and marker sizes in points; font sizes in points.
LINE_WIDTH = 1.5
THIN_LINE_WIDTH = 1.0
GRID_LINE_WIDTH = 0.8
MARKER_SIZE = 3
END_SCORE_MARKER_SIZE = 4
CAP_SIZE = 2
END_SCORE_CAP_SIZE = 3
TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
SMALL_LEGEND_FONT_SIZE = 7


def curve_path(results_dir, run):
    """Path of a run's curve file: one row per (checkpoint, field)."""
    return results_dir / f"{run}_eval_dev_n{EVAL_DEALS}_seed{EVAL_SEED}.csv"


def deals_path(results_dir, run):
    """Path of a run's per-deal file: one row per (checkpoint, field, deal)."""
    return results_dir / f"{run}_eval_dev_n{EVAL_DEALS}_seed{EVAL_SEED}_deals.csv"


def mixed_deals_path(results_dir, run, table, checkpoints):
    """
    Path of a run's per-deal file for one mixed table (a MIXED_TABLES key), scored at
    checkpoints. Matches the name evaluate_checkpoints.py --mixed-table writes.
    """
    checkpoint_names = []
    for checkpoint in checkpoints:
        checkpoint_names.append(str(checkpoint))
    return results_dir / (
        f"{run}_eval_dev_n{EVAL_DEALS}_seed{EVAL_SEED}_{table}{MIXED_TABLES[table]['net_suffix']}"
        f"_ckpt{'-'.join(checkpoint_names)}_deals.csv"
    )


def read_curve(results_dir, run):
    """
    A run's curve: {field: list of rows in training order}, each row a dict with int checkpoint
    and samples and float mean, ci_low, ci_high.
    """
    rows_by_field = {}
    with open(curve_path(results_dir, run)) as file:
        for row in csv.DictReader(file):
            rows_by_field.setdefault(row["field"], []).append({
                "checkpoint": int(row["checkpoint"]),
                "samples": int(row["samples"]),
                "mean": float(row["mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            })
    return rows_by_field


def read_log(runs_dir, run):
    """
    A run's training log: {update: (loss, gradient_norm, cumulative_hours)}.
    Hours = the running sum of wait_seconds + learner_seconds: the learner loop's wall clock,
    without checkpoint saves (small). The config records no start time, so this is the source.
    """
    by_update = {}
    seconds = 0.0
    with open(runs_dir / run / "log.jsonl") as file:
        for line in file:
            entry = json.loads(line)
            seconds += entry["wait_seconds"] + entry["learner_seconds"]
            by_update[entry["update"]] = (
                entry["loss"], entry["gradient_norm"], seconds / SECONDS_PER_HOUR,
            )
    return by_update


def end_checkpoints(curve, field, arm_length):
    """The last END_CHECKPOINTS checkpoints of one field at or before arm_length."""
    checkpoints = []
    for row in curve[field]:
        if row["checkpoint"] <= arm_length:
            checkpoints.append(row["checkpoint"])
    if len(checkpoints) < END_CHECKPOINTS:
        raise ValueError(f"fewer than {END_CHECKPOINTS} {field} checkpoints by {arm_length}")
    return checkpoints[-END_CHECKPOINTS:]


def mean_of(values):
    """The mean of a non-empty list of floats."""
    return sum(values) / len(values)


def load_arms(arms_json, results_dir, runs_dir):
    """
    Reads the arms list and resolves every run (piece 8a).
    Returns (arms, runs, verdicts, control_offsets):
      arms: the parsed arms.json.
      runs: {run: {"curve": read_curve(...), "log": read_log(...) or None}} for every run named.
      verdicts: list of dicts, one per (arm, field): family, label, run, field, verdict, mean,
        ci_low, ci_high. Empty when arm_length_updates is not set yet.
      control_offsets: {field: each control seed's end score minus the 3 seeds' mean}, the
        band plot 5 draws around 0.
    Each MIXED_TABLES field joins verdicts and control_offsets once every control seed has
    that table's file; an arm without one is left out of it. Its label is reported, not ruled on.
    Writes results_dir/arm_verdicts.csv when there are verdicts.
    """
    arms = json.loads(Path(arms_json).read_text())

    # Every run the list names, in a stable order: control seeds, the repeat pair, then arms.
    run_names = list(arms["control"])
    run_names.append(arms["repeat"]["long"])
    run_names.append(arms["repeat"]["short"])
    for family_arms in arms["families"].values():
        for arm in family_arms:
            run_names.append(arm["run"])
            if arm["second_seed_run"]:
                run_names.append(arm["second_seed_run"])

    # Each run's curve, and its training log where one is on this machine.
    runs = {}
    for run in run_names:
        if run in runs:
            continue
        log_file = runs_dir / run / "log.jsonl"
        if log_file.exists():
            log = read_log(runs_dir, run)
        else:
            log = None
        runs[run] = {"curve": read_curve(results_dir, run), "log": log}

    arm_length = arms["arm_length_updates"]
    verdicts = []
    control_offsets = {}
    if arm_length is None:
        return arms, runs, verdicts, control_offsets

    def run_end_scores(run, field):
        # The control's 300M seed-20 run counts at arm length, not at the end of its file.
        if field in MIXED_TABLES:
            checkpoints = end_checkpoints(runs[run]["curve"], FIELDS[0], arm_length)
            return end_scores(mixed_path(run, field), field, checkpoints)
        checkpoints = end_checkpoints(runs[run]["curve"], field, arm_length)
        return end_scores(deals_path(results_dir, run), field, checkpoints)

    def mixed_path(run, table):
        # A mixed file names the same end checkpoints as the run's heuristic curve.
        checkpoints = end_checkpoints(runs[run]["curve"], FIELDS[0], arm_length)
        return mixed_deals_path(results_dir, run, table, checkpoints)

    # A mixed table counts once all 3 control seeds have played it.
    scored_fields = list(FIELDS)
    for table in MIXED_TABLES:
        all_controls_played = True
        for run in arms["control"]:
            if not mixed_path(run, table).exists():
                all_controls_played = False
        if all_controls_played:
            scored_fields.append(table)

    # Per field: the 3 control seeds' end scores, and each one's distance from their mean.
    control_scores_by_field = {}
    for field in scored_fields:
        control_scores = []
        seed_ends = []
        for run in arms["control"]:
            scores = run_end_scores(run, field)
            control_scores.append(scores)
            seed_ends.append(mean_of(scores))
        control_scores_by_field[field] = control_scores
        seed_mean = mean_of(seed_ends)
        offsets = []
        for seed_end in seed_ends:
            offsets.append(seed_end - seed_mean)
        control_offsets[field] = offsets

    # Each arm, on each field it was scored against.
    for family, family_arms in arms["families"].items():
        for arm in family_arms:
            for field in scored_fields:
                if field in MIXED_TABLES:
                    if not mixed_path(arm["run"], field).exists():
                        continue
                elif field not in runs[arm["run"]]["curve"]:
                    continue
                control_scores = control_scores_by_field[field]
                # A named rng per (arm, field) so a label replays.
                rng = random.Random(f"{EVAL_SEED}:{arm['label']}:{field}:bootstrap")
                label, mean, ci_low, ci_high = arm_verdict(
                    run_end_scores(arm["run"], field), control_scores, rng
                )
                # Winner: the second seed passes the same test.
                second_scored = False
                if arm["second_seed_run"]:
                    second_scored = True
                    if field in MIXED_TABLES:
                        if not mixed_path(arm["second_seed_run"], field).exists():
                            second_scored = False
                if label == CANDIDATE and second_scored:
                    second_rng = random.Random(f"{EVAL_SEED}:{arm['label']}:{field}:second")
                    second_label, _, _, _ = arm_verdict(
                        run_end_scores(arm["second_seed_run"], field), control_scores, second_rng
                    )
                    if second_label == CANDIDATE:
                        label = WINNER
                verdicts.append({
                    "family": family, "label": arm["label"], "run": arm["run"], "field": field,
                    "verdict": label, "mean": mean, "ci_low": ci_low, "ci_high": ci_high,
                })

    with open(results_dir / "arm_verdicts.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(verdicts[0]))
        writer.writeheader()
        writer.writerows(verdicts)
    return arms, runs, verdicts, control_offsets


def style(axis, title):
    """The house axis style, as in plot_checkpoint_curve.py."""
    axis.set_title(title, loc="left", fontsize=TITLE_FONT_SIZE)
    axis.grid(color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=TICK_FONT_SIZE)


def x_values(rows, log, x_axis):
    """
    Each row's x: millions of samples, or the learner's hours at that checkpoint's update.
    rows: curve rows; log: read_log's dict (needed for hours); x_axis: "samples" or "hours".
    """
    values = []
    for row in rows:
        if x_axis == "samples":
            values.append(row["samples"] / SAMPLES_PER_MILLION)
        elif row["checkpoint"] == 0:
            values.append(0.0)
        else:
            values.append(log[row["checkpoint"]][2])
    return values


def plot_curves(arms, runs, field, x_axis, plot_path):
    """
    Plots 1-3 (piece 8b): score against one field by samples or by hours, one panel per family.
    Each panel: the control band (min to max of the 3 seeds at each checkpoint, to arm length)
    with their mean, and each arm's curve with its deal-interval bars.
    field: "heuristic" or "greedy"; x_axis: "samples" or "hours".
    """
    arm_length = arms["arm_length_updates"]
    families = list(arms["families"])
    figure, axes = plt.subplots(1, len(families),
                                figsize=(PANEL_WIDTH * len(families), PANEL_HEIGHT),
                                squeeze=False, sharey=True)

    # Each control seed's rows up to arm length.
    control_rows = []
    for run in arms["control"]:
        seed_rows = []
        for row in runs[run]["curve"][field]:
            if row["checkpoint"] <= arm_length:
                seed_rows.append(row)
        control_rows.append(seed_rows)

    # The band covers the checkpoints all 3 seeds share: min, max and mean of their scores.
    num_points = len(control_rows[0])
    for seed_rows in control_rows:
        num_points = min(num_points, len(seed_rows))
    first_seed = arms["control"][0]
    band_x = x_values(control_rows[0][:num_points], runs[first_seed]["log"], x_axis)
    band_low = []
    band_high = []
    band_mean = []
    for index in range(num_points):
        means = []
        for seed_rows in control_rows:
            means.append(seed_rows[index]["mean"])
        band_low.append(min(means))
        band_high.append(max(means))
        band_mean.append(mean_of(means))

    if x_axis == "samples":
        x_label = "samples (millions)"
    else:
        x_label = "learner hours"

    for axis, family in zip(axes[0], families):
        axis.fill_between(band_x, band_low, band_high, color=CONTROL_COLOR, alpha=BAND_ALPHA,
                          linewidth=0, label="control, 3 seeds")
        axis.plot(band_x, band_mean, color=CONTROL_COLOR, linewidth=LINE_WIDTH)
        for arm in arms["families"][family]:
            curve = runs[arm["run"]]["curve"]
            if field not in curve:
                continue
            # Each checkpoint's mean, and its interval as distances below and above the mean.
            means = []
            below = []
            above = []
            for row in curve[field]:
                means.append(row["mean"])
                below.append(row["mean"] - row["ci_low"])
                above.append(row["ci_high"] - row["mean"])
            arm_x = x_values(curve[field], runs[arm["run"]]["log"], x_axis)
            axis.errorbar(arm_x, means, yerr=[below, above], marker="o",
                          markersize=MARKER_SIZE, linewidth=LINE_WIDTH, capsize=CAP_SIZE,
                          label=arm["label"])
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=THIN_LINE_WIDTH, linestyle="--")
        axis.set_xlabel(x_label)
        style(axis, family)
        axis.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="lower right")
    axes[0][0].set_ylabel(f"mean points per deal vs {field} (95% CI over deals)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(figure)


def plot_training(arms, runs, plot_path):
    """
    Plot 4 (piece 8c): training loss and gradient norm by update, one line per run with a log,
    each a mean over blocks of LOG_BLOCK updates. Control seeds are drawn in the band's gray.
    """
    figure, (loss_axis, norm_axis) = plt.subplots(1, 2, figsize=TRAINING_FIGURE_SIZE)

    # The name each run is drawn under: control runs by run name, arms by their label.
    names = {}
    for run in arms["control"]:
        names[run] = run
    for family_arms in arms["families"].values():
        for arm in family_arms:
            names[arm["run"]] = arm["label"]

    for run, name in names.items():
        log = runs[run]["log"]
        if log is None:
            continue
        # Block means: x is the block's last update.
        updates = sorted(log)
        block_x = []
        block_loss = []
        block_norm = []
        for start in range(0, len(updates), LOG_BLOCK):
            block = updates[start:start + LOG_BLOCK]
            losses = []
            norms = []
            for update in block:
                losses.append(log[update][0])
                norms.append(log[update][1])
            block_x.append(block[-1])
            block_loss.append(mean_of(losses))
            block_norm.append(mean_of(norms))
        # None: matplotlib picks the arm's color from its default cycle.
        if run in arms["control"]:
            color = CONTROL_COLOR
        else:
            color = None
        loss_axis.plot(block_x, block_loss, linewidth=THIN_LINE_WIDTH, color=color, label=name)
        norm_axis.plot(block_x, block_norm, linewidth=THIN_LINE_WIDTH, color=color, label=name)

    loss_axis.set_ylabel("training loss (MSE)")
    norm_axis.set_ylabel("gradient norm (before clipping)")
    for axis, title in ((loss_axis, "loss"), (norm_axis, "gradient norm")):
        axis.set_xlabel("update")
        style(axis, title)
    norm_axis.legend(frameon=False, fontsize=SMALL_LEGEND_FONT_SIZE)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(figure)


def plot_end_scores(verdicts, control_offsets, plot_path, field="heuristic"):
    """
    Plot 5 (piece 8d): every arm's [arm - control mean] with its 95% interval over deals, one
    row per arm, the label written beside it. 0 is the control's mean; the 3 control seeds are
    dotted lines, so "above every seed" can be read off.
    verdicts, control_offsets: from load_arms.
    """
    rows = []
    for verdict in verdicts:
        if verdict["field"] == field:
            rows.append(verdict)
    height = END_SCORES_BASE_HEIGHT + END_SCORES_ROW_HEIGHT * len(rows)
    figure, axis = plt.subplots(figsize=(END_SCORES_WIDTH, height))

    tick_labels = []
    for index, row in enumerate(rows):
        below = row["mean"] - row["ci_low"]
        above = row["ci_high"] - row["mean"]
        axis.errorbar([row["mean"]], [index], xerr=[[below], [above]], marker="o",
                      markersize=END_SCORE_MARKER_SIZE, capsize=END_SCORE_CAP_SIZE,
                      color=TEXT_SECONDARY)
        axis.annotate(row["verdict"], (row["ci_high"], index), xytext=(LABEL_OFFSET_POINTS, 0),
                      textcoords="offset points", va="center", fontsize=TICK_FONT_SIZE)
        tick_labels.append(f"{row['family']}: {row['label']}")

    axis.axvline(0, color=CONTROL_COLOR, linewidth=LINE_WIDTH)
    for offset in control_offsets[field]:
        axis.axvline(offset, color=CONTROL_COLOR, linewidth=THIN_LINE_WIDTH, linestyle=":")
    axis.margins(x=END_SCORES_X_MARGIN)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(tick_labels, fontsize=TICK_FONT_SIZE)
    axis.set_xlabel(f"arm - control mean, points per deal vs {field} (95% CI over deals)")
    if field in MIXED_TABLES:
        style(axis, MIXED_TABLES[field]["title"])
    else:
        style(axis, "end scores against the control (last 3 checkpoints at arm length)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(figure)


def plot_repeat(arms, runs, plot_path):
    """
    Plot 6 (piece 8e): the 300M seed-20 curve over stage 4's 60M seed-20 curve, every field both
    were scored on. Same seed, same arguments: the gap is a read on same-seed repeatability.
    The 300M run is solid, the 60M run dashed.
    """
    figure, axis = plt.subplots(figsize=REPEAT_FIGURE_SIZE)
    for role, line_style in (("long", "-"), ("short", "--")):
        run = arms["repeat"][role]
        for field, rows in runs[run]["curve"].items():
            millions = []
            means = []
            lows = []
            highs = []
            for row in rows:
                millions.append(row["samples"] / SAMPLES_PER_MILLION)
                means.append(row["mean"])
                lows.append(row["ci_low"])
                highs.append(row["ci_high"])
            axis.plot(millions, means, linestyle=line_style, marker="o", markersize=MARKER_SIZE,
                      linewidth=LINE_WIDTH, label=f"{run} vs {field}")
            axis.fill_between(millions, lows, highs, alpha=BAND_ALPHA, linewidth=0)
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=THIN_LINE_WIDTH, linestyle="--")
    axis.set_xlabel("samples (millions)")
    axis.set_ylabel("mean points per deal over the field (95% CI over deals)")
    style(axis, "same seed, same arguments: 300M run over stage 4's 60M run")
    axis.legend(frameon=False, fontsize=SMALL_LEGEND_FONT_SIZE, loc="lower right")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(figure)


def make_all(arms_json, results_dir, runs_dir):
    """
    The labels and every plot the arms list supports: plot 6 always; the labels and plots 1-5
    once arm_length_updates is set. Returns the Paths written.
    """
    arms, runs, verdicts, control_offsets = load_arms(arms_json, results_dir, runs_dir)
    written = []
    path = results_dir / "arms_6_repeat.png"
    plot_repeat(arms, runs, path)
    written.append(path)
    if not verdicts:
        return written
    written.append(results_dir / "arm_verdicts.csv")
    for path, field, x_axis in (
        (results_dir / "arms_1_heuristic_by_samples.png", "heuristic", "samples"),
        (results_dir / "arms_2_greedy_by_samples.png", "greedy", "samples"),
        (results_dir / "arms_3_heuristic_by_hours.png", "heuristic", "hours"),
    ):
        plot_curves(arms, runs, field, x_axis, path)
        written.append(path)
    path = results_dir / "arms_4_training.png"
    plot_training(arms, runs, path)
    written.append(path)
    path = results_dir / "arms_5_end_scores.png"
    plot_end_scores(verdicts, control_offsets, path)
    written.append(path)
    # Plots 7 and 8: the same end scores on each mixed table the control seeds have all played.
    for table, table_settings in MIXED_TABLES.items():
        if table in control_offsets:
            path = results_dir / table_settings["plot"]
            plot_end_scores(verdicts, control_offsets, path, field=table)
            written.append(path)
    return written


def main():
    """Parses the command line and writes the labels and plots."""
    parser = argparse.ArgumentParser(description="Stage 5 arm labels and plots.")
    parser.add_argument("--arms", type=Path, default=RESULTS_DIR / "arms.json")
    args = parser.parse_args()
    for path in make_all(args.arms, RESULTS_DIR, RUNS_DIR):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
