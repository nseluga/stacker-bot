"""Checks src/stacker/characterize.py (stage plan characterization)."""

import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from stacker.agents import greedy_agent
from stacker.characterize import (
    counting_agent,
    deal_luck_share,
    deals_needed,
    decisions_per_second,
    game_shape,
    play_for,
)
from stacker.game import PASS, new_hand, play_out

# Any dev seed ≥ 10 (rule 2); the test only needs one fixed hand.
TEST_SEED = 10
NUM_PLAYERS = 4


def play_seeded_hand(agent):
    """Deals and plays the TEST_SEED hand with agent in every seat; returns (trace, points)."""
    rng = random.Random(TEST_SEED)
    state = new_hand(NUM_PLAYERS, rng)
    agents = []
    for _ in range(NUM_PLAYERS):
        agents.append(agent)
    return play_out(state, agents, rng)


def test_counting_agent_records_every_decision_without_changing_play():
    log = []
    trace, points = play_seeded_hand(counting_agent(greedy_agent, log))
    plain_trace, plain_points = play_seeded_hand(greedy_agent)

    # The wrapper only watches: the same seed gives the same hand.
    assert trace == plain_trace
    assert points == plain_points

    # One record per action in the trace, and each passed flag matches the trace.
    assert len(log) == len(trace)
    for decision, (_seat, action) in zip(log, trace):
        assert decision.passed == (action == PASS)
        assert decision.legal_count >= 1
        # A pass can only happen where one was allowed.
        if decision.passed:
            assert decision.pass_allowed

    # The opening play is a lead, so no pass was allowed.
    assert log[0].pass_allowed is False


SAMPLE_HANDS = 20


def test_game_shape_counts_add_up_and_replay():
    hand_rows, summary = game_shape(NUM_PLAYERS, greedy_agent, SAMPLE_HANDS, TEST_SEED)
    assert len(hand_rows) == SAMPLE_HANDS

    # The pass rate is a share of pass chances, and passes never outnumber chances in a hand.
    assert 0 <= summary["pass_rate"] <= 1
    for row in hand_rows:
        assert row["passes"] <= row["pass_chances"] < row["moves"]

    # The same seed gives the same numbers.
    assert game_shape(NUM_PLAYERS, greedy_agent, SAMPLE_HANDS, TEST_SEED) == (hand_rows, summary)

    # Hand 0 replayed by hand with counting_agent gives the first row.
    log = []
    counted = counting_agent(greedy_agent, log)
    rng = random.Random(f"{TEST_SEED}:0")
    play_out(new_hand(NUM_PLAYERS, rng), [counted] * NUM_PLAYERS, rng)
    passes = 0
    forced = 0
    for decision in log:
        if decision.passed:
            passes += 1
        if decision.legal_count == 1:
            forced += 1
    assert hand_rows[0]["moves"] == len(log)
    assert hand_rows[0]["passes"] == passes
    assert hand_rows[0]["forced"] == forced

    # Greedy passes only when PASS is its one action, so it never passes by choice.
    assert summary["choice_pass_rate"] == 0
    assert 0 < summary["forced_share"] < 1
    assert summary["moves_p90"] <= summary["moves_p99"] <= summary["moves_max"]


def run_characterize_game(arguments):
    """Runs scripts/characterize_game.py as a user would; returns the finished process."""
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = [sys.executable, str(repo_root / "scripts" / "characterize_game.py")]
    command.extend(arguments)
    return subprocess.run(command, cwd=repo_root, env=environment, capture_output=True, text=True)


def test_characterize_game_script_writes_the_game_shape(tmp_path):
    script_seed = 13
    script_hands = 5
    finished = run_characterize_game(
        ["--players", "4", "--agent", "greedy", "--hands", str(script_hands),
         "--seed", str(script_seed), "--out-dir", str(tmp_path)]
    )
    assert finished.returncode == 0, finished.stderr

    # One CSV row per hand, and the JSON summary is exactly game_shape's.
    run_name = f"4p1d_game-shape_greedy_dev_n{script_hands}_seed{script_seed}"
    with open(tmp_path / (run_name + ".csv")) as file:
        assert len(list(csv.DictReader(file))) == script_hands
    with open(tmp_path / (run_name + ".json")) as file:
        record = json.load(file)
    _rows, summary = game_shape(NUM_PLAYERS, greedy_agent, script_hands, script_seed)
    assert record["summary"] == summary


def test_characterize_game_script_refuses_a_test_seed(tmp_path):
    # Rule 2: seed 2 is a test seed, so the run stops before playing anything.
    finished = run_characterize_game(
        ["--players", "4", "--agent", "greedy", "--hands", "5", "--seed", "2",
         "--out-dir", str(tmp_path)]
    )
    assert finished.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_deal_luck_share_edge_cases_and_hand_value():
    # Constant-sum rotations: the deal only moves points between seats, so pairing cancels all.
    share, sigma, _seat, _deal = deal_luck_share([[2, 0, 1, 0], [0, 0, 1, 2]])
    assert share == 1
    assert sigma == 0

    # Identical rotations: pairing gains nothing, share = 1 - num_players.
    share, _sigma, _seat, _deal = deal_luck_share([[2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]])
    assert abs(share - (1 - 4)) < 1e-12

    # Hand-computed: pooled pvariance 0.359375 / 4 = 0.08984375; means 0.75, 1.0 -> 0.015625.
    share, _sigma, _seat, _deal = deal_luck_share([[2, 0, 0, 1], [1, 1, 1, 1]])
    assert abs(share - 19 / 23) < 1e-12

    # Every score equal: 0 / 0, reported as None.
    assert deal_luck_share([[1, 1], [1, 1]]) == (None, 0, 0, 0)


def test_deal_luck_share_splits_the_pooled_variance():
    # Law of total variance: seat luck + deal luck = pooled variance, on any rows.
    rng = random.Random(TEST_SEED)
    rotation_rows = []
    for _ in range(50):
        scores = []
        for _rotation in range(NUM_PLAYERS):
            scores.append(rng.choice((0, 1, 2, 3)))
        rotation_rows.append(scores)
    _share, _sigma, seat_variance, deal_variance = deal_luck_share(rotation_rows)
    pooled = []
    for scores in rotation_rows:
        pooled.extend(scores)
    assert abs(seat_variance + deal_variance - statistics.pvariance(pooled)) < 1e-12

    # Hand-computed: deal [2, 0, 0, 1] has variance 0.6875, deal [1, 1, 1, 1] has 0.
    _share, _sigma, seat_variance, deal_variance = deal_luck_share([[2, 0, 0, 1], [1, 1, 1, 1]])
    assert abs(seat_variance - 0.34375) < 1e-12
    assert abs(deal_variance - 0.015625) < 1e-12


def run_deal_luck(arguments):
    """Runs scripts/deal_luck.py as a user would; returns the finished process."""
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = [sys.executable, str(repo_root / "scripts" / "deal_luck.py")]
    command.extend(arguments)
    return subprocess.run(command, cwd=repo_root, env=environment, capture_output=True, text=True)


def test_deal_luck_script_writes_share_from_its_rows(tmp_path):
    finished = run_deal_luck(
        ["--players", "4", "--test", "greedy", "--field", "random", "--deals", "5",
         "--seed", "13", "--out-dir", str(tmp_path)]
    )
    assert finished.returncode == 0, finished.stderr

    # The JSON share and σ are deal_luck_share on exactly the CSV's rows.
    run_name = "4p1d_deal-luck_greedy-vs-random_dev_n5_seed13"
    rotation_rows = []
    with open(tmp_path / (run_name + ".csv")) as file:
        for row in csv.DictReader(file):
            scores = []
            for rotation in range(NUM_PLAYERS):
                scores.append(int(row[f"rotation_{rotation}"]))
            rotation_rows.append(scores)
    assert len(rotation_rows) == 5
    with open(tmp_path / (run_name + ".json")) as file:
        record = json.load(file)
    expected = (
        record["deal_luck_share"],
        record["standard_deviation"],
        record["seat_variance"],
        record["deal_variance"],
    )
    assert expected == deal_luck_share(rotation_rows)


def test_deal_luck_script_refuses_a_test_seed(tmp_path):
    # Rule 2: seed 2 is a test seed, so the run stops before playing anything.
    finished = run_deal_luck(
        ["--players", "4", "--test", "greedy", "--field", "random", "--deals", "5",
         "--seed", "2", "--out-dir", str(tmp_path)]
    )
    assert finished.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_deals_needed_matches_the_m1_confirm():
    # Decision log "M1 confirm uses δ = 0.10": pilot σ 0.366 gives (2.8·0.366/0.10)² = 105.02,
    # logged as ≈ 105; rounded up, 106 deals.
    assert deals_needed(0.366, 0.10) == 106
    # Halving δ quadruples N (before rounding up).
    assert deals_needed(1.0, 0.5) == 32
    assert deals_needed(1.0, 0.25) == 126


def test_play_for_gives_each_worker_its_own_hands():
    # seconds = 0 plays exactly one hand: hand 0 of that worker, dealt from f"{seed}:{worker}:0".
    for worker in (0, 1):
        decisions, hands, _elapsed = play_for(NUM_PLAYERS, greedy_agent, 0, TEST_SEED, worker)
        rng = random.Random(f"{TEST_SEED}:{worker}:0")
        trace, _points = play_out(new_hand(NUM_PLAYERS, rng), [greedy_agent] * NUM_PLAYERS, rng)
        assert hands == 1
        assert decisions == len(trace)


def test_decisions_per_second_sums_two_workers():
    decision_rate, hand_rate, per_worker = decisions_per_second(
        NUM_PLAYERS, greedy_agent, 0.5, 2, TEST_SEED
    )
    assert len(per_worker) == 2
    assert decision_rate > hand_rate > 0


def run_throughput(arguments):
    """Runs scripts/throughput.py as a user would; returns the finished process."""
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = [sys.executable, str(repo_root / "scripts" / "throughput.py")]
    command.extend(arguments)
    return subprocess.run(command, cwd=repo_root, env=environment, capture_output=True, text=True)


def test_throughput_script_writes_one_row_per_process_count(tmp_path):
    finished = run_throughput(
        ["--players", "4", "--agent", "greedy", "--seconds", "0.3", "--processes", "1,2",
         "--seed", "13", "--out-dir", str(tmp_path)]
    )
    assert finished.returncode == 0, finished.stderr
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    with open(written[0]) as file:
        record = json.load(file)
    process_counts = []
    for row in record["rows"]:
        process_counts.append(row["processes"])
    assert process_counts == [1, 2]
    assert len(record["rows"][1]["workers"]) == 2
    # Context fields: load and wall time per row; usable CPUs is None off Linux.
    assert record["rows"][0]["wall_seconds"] >= 0.3
    assert record["rows"][0]["load_one_minute_before"] >= 0
    assert "usable_cpus" in record
    # Without --repeats every row is measured once; greedy has no learner to time.
    for row in record["rows"]:
        assert row["repeat"] == 1
    assert record["learner"] == []
    assert record["net_weights"] is None


def test_throughput_script_times_the_net_and_the_learner_per_repeat(tmp_path):
    # About 6 s: the learner timing first self-plays one full batch of samples.
    finished = run_throughput(
        ["--players", "4", "--agent", "net", "--seconds", "0.3", "--processes", "1",
         "--seed", "13", "--repeats", "2", "--out-dir", str(tmp_path)]
    )
    assert finished.returncode == 0, finished.stderr
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    with open(written[0]) as file:
        record = json.load(file)

    # The script stops before timing if a worker gets a copy of the weights, so a written
    # record means its own sharing check passed.
    assert record["net_weights"] == "shared"

    # One row per repeat of the one process count, numbered from 1.
    repeats = []
    for row in record["rows"]:
        assert row["decisions_per_second"] > 0
        repeats.append(row["repeat"])
    assert repeats == [1, 2]

    learner_repeats = []
    for entry in record["learner"]:
        assert entry["samples_per_second"] > 0
        assert entry["updates"] >= 1
        assert entry["torch_threads"] >= 1
        learner_repeats.append(entry["repeat"])
    assert learner_repeats == [1, 2]


def test_throughput_script_refuses_a_test_seed(tmp_path):
    # Rule 2: seed 2 is a test seed, so the run stops before playing anything.
    finished = run_throughput(
        ["--players", "4", "--agent", "greedy", "--seconds", "0.3", "--processes", "1",
         "--seed", "2", "--out-dir", str(tmp_path)]
    )
    assert finished.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_sigma_to_n_script_names_its_table_after_the_run(tmp_path):
    # A minimal deal-luck JSON: the script reads only standard_deviation.
    sigma = 0.5
    deal_luck_json = tmp_path / "4p1d_deal-luck_greedy-vs-random_dev_n5_seed13.json"
    with open(deal_luck_json, "w") as file:
        json.dump({"players": 4, "standard_deviation": sigma}, file)

    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = [sys.executable, str(repo_root / "scripts" / "sigma_to_n.py"), str(deal_luck_json),
               "--out-dir", str(tmp_path)]
    finished = subprocess.run(command, cwd=repo_root, env=environment, capture_output=True,
                              text=True)
    assert finished.returncode == 0, finished.stderr

    # One row per δ, each deals_needed(σ, δ), in a table named after the pair and run.
    with open(tmp_path / "4p1d_sigma-to-n_greedy-vs-random_dev_n5_seed13.csv") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    for row in rows:
        assert int(row["deals_needed"]) == deals_needed(sigma, float(row["delta"]))
