"""
Gates for scripts/throughput_ceiling.py (stage plan specialist_explore, stage 5 piece 13): the
batched action pick matches choose_action decision by decision, and the projected gain's
arithmetic on one hand-worked case.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import throughput_ceiling
from test_net import DEV_SEED

from stacker.dmc import choose_action

PLAYERS = 4
NUM_DECISIONS = 16
# Pushes every real Q far below 0, so a padded slot (which holds exactly 0) would win any argmax
# that forgot the action mask.
NEGATIVE_BIAS = -100.0

# The hand-worked case for projected_gain: 100 decisions, half of them choice turns, 10 s in all.
# Today: 10 / 100 = 0.1 s per decision. Batched: (2 + 2) / 100 kept, plus 0.01 s on each choice
# turn = 0.04 + 0.005 = 0.045 s. Actor speedup 0.1 / 0.045 = 20 / 9.
SPLIT = {
    "wall_seconds": 10.0,
    "encode_seconds": 2.0,
    "choose_seconds": 6.0,
    "forward_seconds": 5.0,
    "engine_and_other_seconds": 2.0,
    "decisions": 100,
    "choice_turns": 50,
}
BATCHED_SECONDS_PER_DECISION = 0.01
EXPECTED_ACTOR_SPEEDUP = 20.0 / 9.0
# Below the actor speedup, so the run speedup must stop at it.
LOW_CAP = 1.5


def test_batched_choices_match_choose_action_per_decision():
    # catches: a padded action winning the argmax (every real Q is pushed below 0), and one
    # decision getting another's answer (argmax along the wrong axis, or rows out of order).
    decisions = throughput_ceiling.choice_decisions(PLAYERS, DEV_SEED, NUM_DECISIONS)
    action_counts = set()
    for decision in decisions:
        action_counts.add(len(decision[3]))
    # With one K only, nothing is padded and the padded-slot check proves nothing.
    assert len(action_counts) > 1

    net = throughput_ceiling.untrained_net()
    last_layer = net.mlp[-1]
    with torch.no_grad():
        last_layer.bias.fill_(NEGATIVE_BIAS)

    batched = throughput_ceiling.batched_choices(net, decisions, torch.device("cpu"))
    one_by_one = []
    for decision in decisions:
        # epsilon 0: choose_action never explores, so the rng is never drawn from.
        one_by_one.append(choose_action(net, decision, 0.0, random.Random(DEV_SEED)))
    assert batched == one_by_one


def test_projected_gain_hand_worked_case():
    # catches: the speedup inverted (would be 0.45), forced turns charged the batched cost
    # (would be 2.0), and the learner cap not applied (the run speedup would be 20 / 9).
    actor_speedup, run_speedup = throughput_ceiling.projected_gain(
        SPLIT, BATCHED_SECONDS_PER_DECISION, LOW_CAP
    )
    assert actor_speedup == pytest.approx(EXPECTED_ACTOR_SPEEDUP)
    assert run_speedup == LOW_CAP
