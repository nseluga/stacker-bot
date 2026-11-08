"""Checks src/stacker/characterize.py (stage plan characterization)."""

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from stacker.agents import greedy_agent
from stacker.characterize import (
    counting_agent,
    deal_luck_share,
    deals_needed,
    game_shape,
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
    for decision in log:
        if decision.passed:
            passes += 1
    assert hand_rows[0]["moves"] == len(log)
    assert hand_rows[0]["passes"] == passes


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
    share, sigma = deal_luck_share([[2, 0, 1, 0], [0, 0, 1, 2]])
    assert share == 1
    assert sigma == 0

    # Identical rotations: pairing gains nothing, share = 1 - num_players.
    share, _sigma = deal_luck_share([[2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]])
    assert abs(share - (1 - 4)) < 1e-12

    # Hand-computed: pooled pvariance 0.359375 / 4 = 0.08984375; means 0.75, 1.0 -> 0.015625.
    share, _sigma = deal_luck_share([[2, 0, 0, 1], [1, 1, 1, 1]])
    assert abs(share - 19 / 23) < 1e-12

    # Every score equal: 0 / 0, reported as None.
    assert deal_luck_share([[1, 1], [1, 1]]) == (None, 0)


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
    assert (record["deal_luck_share"], record["standard_deviation"]) == deal_luck_share(
        rotation_rows
    )


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
