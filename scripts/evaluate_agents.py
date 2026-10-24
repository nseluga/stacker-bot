"""
Paired duplicate-deal evaluation of one agent against a field (architecture plan §3 rule 4;
stage plan harness, piece 10).

Plays every deal once per seat rotation with the test agent in --test-seats (shifted each
rotation) and the field agent elsewhere. Then it writes the per-deal paired differences and a
percentile-bootstrap 95% CI to results/<slug>/.

Two deal sources:
- dev (default): fresh shuffles from an integer seed ≥ 10, regenerated each run, never stored.
- sealed (--confirm): the first N deals of data/sealed-deals/<config>/, hash-checked on load.
  Only a pre-registered confirm run uses them (rule 1), and only with a test seed 0–2 (rule 2).

Run from the repo root, e.g. random vs random, 4 players, 1v3, 200 dev deals:
    PYTHONPATH=src .venv/bin/python scripts/evaluate_agents.py --players 4 --test random \
        --field random --test-seats 0 --deals 200 --seed 10 --slug harness
Output: results/<slug>/<run name>.csv (deal_index, difference) and .json (arguments, mean, CI,
standard deviation, git commit, timings). The run name holds every argument, so runs never collide.
"""

import argparse
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from stacker.agents import random_agent
from stacker.cards import CONFIGS, build_deck
from stacker.evaluate import bootstrap_ci, paired_differences
from stacker.sealed import config_name, load_sealed

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

# Agents a run can name. greedy and heuristic join at the baselines stage.
AGENTS = {"random": random_agent}

# Rule 2: seeds 0–2 are test seeds, for confirm runs only. Exploration uses seeds from 10 up.
TEST_SEEDS = (0, 1, 2)
DEV_SEED_MIN = 10


def select_deals(num_players, count, seed, confirm):
    """
    The deck orders to evaluate on, in deal_index order.
    num_players: 3 to 8. count: number of deals, at least 1. seed: int.
    confirm: False for dev deals, True for the sealed set.
    Dev: one random.Random(seed) shuffles count decks in turn, so the first k deals are the same
    for any count (a short probe sees the start of the long run's deals).
    Sealed: the first count deals of the config's sealed file; load_sealed raises if its hash fails.
    Raises ValueError when the seed breaks rule 2 or count is out of range.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    # Rule 2, enforced here so no run can mix test seeds with dev deals or the reverse.
    if confirm:
        if seed not in TEST_SEEDS:
            raise ValueError(f"a confirm run uses a test seed {TEST_SEEDS}, got {seed}")
    else:
        if seed < DEV_SEED_MIN:
            raise ValueError(f"dev runs use seeds ≥ {DEV_SEED_MIN}; 0–2 are test seeds, got {seed}")

    if confirm:
        sealed = load_sealed(num_players)
        if count > len(sealed):
            raise ValueError(f"{count} deals asked, the sealed set holds {len(sealed)}")
        return sealed[:count]

    rng = random.Random(seed)
    deck_orders = []
    for _ in range(count):
        deck_order = build_deck(CONFIGS[num_players].num_decks)
        rng.shuffle(deck_order)
        deck_orders.append(deck_order)
    return deck_orders


def git_state():
    """The commit the code ran at, and whether uncommitted changes could make it not reproduce."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return commit, changes != ""


def main():
    """Parses the command line, plays the deals, and writes the per-deal CSV and summary JSON."""
    parser = argparse.ArgumentParser(description="Paired duplicate-deal evaluation, test vs field.")
    parser.add_argument("--players", type=int, required=True, help="3 to 8")
    parser.add_argument("--test", choices=sorted(AGENTS), required=True)
    parser.add_argument("--field", choices=sorted(AGENTS), required=True)
    parser.add_argument("--test-seats", required=True, help="seats at rotation 0, e.g. 0 or 0,2")
    parser.add_argument("--deals", type=int, required=True, help="number of deals N")
    parser.add_argument("--seed", type=int, required=True, help="dev: ≥ 10; confirm: 0, 1 or 2")
    parser.add_argument("--slug", required=True, help="results/<slug>/, a manifest slug")
    parser.add_argument("--confirm", action="store_true", help="sealed deals; pre-registered runs only")
    args = parser.parse_args()

    test_seats = []
    seat_labels = []
    for part in args.test_seats.split(","):
        test_seats.append(int(part))
        seat_labels.append(part.strip())
    test_seats = tuple(test_seats)

    if args.confirm:
        source = "sealed"
    else:
        source = "dev"
    run_name = (
        f"{config_name(args.players)}_{args.test}-vs-{args.field}_seats{'-'.join(seat_labels)}"
        f"_{source}_n{args.deals}_seed{args.seed}"
    )
    out_dir = RESULTS_ROOT / args.slug
    csv_path = out_dir / (run_name + ".csv")
    json_path = out_dir / (run_name + ".json")

    # A confirm is a one-shot event: refuse to replace one before any deal is loaded or played.
    if args.confirm and (csv_path.exists() or json_path.exists()):
        raise SystemExit(f"{csv_path.stem} already has results; a confirm run is never repeated")

    deck_orders = select_deals(args.players, args.deals, args.seed, args.confirm)

    start = time.perf_counter()
    differences = paired_differences(
        deck_orders, args.players, AGENTS[args.test], AGENTS[args.field], test_seats, args.seed
    )
    play_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mean, low, high = bootstrap_ci(differences, random.Random(f"{args.seed}:bootstrap"))
    bootstrap_seconds = time.perf_counter() - start

    # σ of the per-deal differences: the pilot's input to the power analysis that sets N.
    if len(differences) > 1:
        standard_deviation = statistics.stdev(differences)
    else:
        standard_deviation = None

    commit, uncommitted_changes = git_state()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["deal_index", "difference"])
        for deal_index, difference in enumerate(differences):
            writer.writerow([deal_index, difference])

    summary = {
        "players": args.players,
        "test": args.test,
        "field": args.field,
        "test_seats": list(test_seats),
        "deals": args.deals,
        "seed": args.seed,
        "source": source,
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "standard_deviation": standard_deviation,
        "bootstrap_resamples_seed": f"{args.seed}:bootstrap",
        "git_commit": commit,
        "uncommitted_changes": uncommitted_changes,
        "play_seconds": round(play_seconds, 2),
        "bootstrap_seconds": round(bootstrap_seconds, 2),
    }
    with open(json_path, "w") as file:
        json.dump(summary, file, indent=2)

    print(
        f"{run_name}: mean {mean:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}], "
        f"sd {standard_deviation}, play {play_seconds:.1f} s, bootstrap {bootstrap_seconds:.1f} s"
    )


if __name__ == "__main__":
    main()
