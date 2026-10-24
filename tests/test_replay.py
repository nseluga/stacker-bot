"""
Tests for deterministic replay (stage plan engine_play, tracker step "Deterministic replay").

One seed drives the deal and every agent choice, so a hand must replay exactly from its seed.
The same seed twice gives the identical trace and points; a different seed gives a different
trace. A second check replays in fresh Python processes with different hash seeds, because a
sealed deal set (plan §3) must replay the same on any run, not just inside one process.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import choose_random
from stacker.cards import CONFIGS
from stacker.game import PASS, play_hand

REPLAY_SEED = 10
OTHER_SEED = 11

# Two different string-hash seeds for the cross-process check.
HASH_SEEDS = ("1", "2")

REPO_ROOT = Path(__file__).parent.parent


def trace_text(trace):
    """Writes a trace as one line of text, so traces from separate processes can be compared."""
    steps = []
    for seat, action in trace:
        if action == PASS:
            steps.append(f"{seat}:pass")
        else:
            names = []
            for card in action.cards:
                names.append(str(card))
            steps.append(f"{seat}:" + " ".join(names))
    return " | ".join(steps)


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_same_seed_replays_the_same_hand(num_players):
    first = play_hand(num_players, random.Random(REPLAY_SEED), choose_random)
    second = play_hand(num_players, random.Random(REPLAY_SEED), choose_random)
    assert first == second


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_different_seed_plays_a_different_hand(num_players):
    first_trace, _ = play_hand(num_players, random.Random(REPLAY_SEED), choose_random)
    other_trace, _ = play_hand(num_players, random.Random(OTHER_SEED), choose_random)
    assert first_trace != other_trace


# The child process plays one hand per config and prints each trace on its own line.
CHILD_SCRIPT = """
import random
from helpers import choose_random
from stacker.cards import CONFIGS
from stacker.game import play_hand
from test_replay import REPLAY_SEED, trace_text
for num_players in CONFIGS:
    trace, scores = play_hand(num_players, random.Random(REPLAY_SEED), choose_random)
    print(trace_text(trace), scores)
"""


def test_replay_is_the_same_across_processes_and_hash_seeds():
    outputs = []
    for hash_seed in HASH_SEEDS:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT / "tests")
        result = subprocess.run(
            [sys.executable, "-c", CHILD_SCRIPT],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert len(outputs[0].splitlines()) == len(CONFIGS)
