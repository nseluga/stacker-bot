"""
Gates for src/stacker/arms.py (stage plan specialist_explore, stage 5 pieces 7a-7b): end scores
read from a deals file, and the arm rule's three known-answer cases.
"""

import csv
import random

from test_net import DEV_SEED

from stacker.arms import CANDIDATE, LOSES, TIE, arm_verdict, end_scores

# Enough deals for a tight interval; the bootstrap stays under a second.
NUM_DEALS = 200
# The control seeds sit at fixed offsets from one another, per deal: a band 0.2 wide.
CONTROL_OFFSETS = (0.0, 0.1, 0.2)
# The hand-written deals file: 4 checkpoints, 2 deals, greedy rows at a value far from any mean.
FILE_CHECKPOINTS = (0, 10, 20, 30)
NAMED_CHECKPOINTS = [10, 20, 30]
NUM_FILE_DEALS = 2
DEAL_STEP = 100
OTHER_FIELD_VALUE = -1000.0
# Per-deal spread shared by every run (deal luck), near the measured per-deal SD (0.23-0.36).
DEAL_SPREAD = 0.3
# Each arm's own per-deal noise: small, so the interval has width but stays tight.
ARM_NOISE = 0.01
# Arm offsets for the three known answers: above the best seed, below the worst seed, and
# above the band's mean (0.1) but below the best seed (0.2).
CLEAR_WIN_OFFSET = 0.3
CLEAR_LOSS_OFFSET = -0.1
BETWEEN_OFFSET = 0.15


def test_end_scores_average_the_named_checkpoints_of_one_field(tmp_path):
    # catches: averaging every checkpoint in the file, mixing fields, or averaging across deals.
    path = tmp_path / "run_deals.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["checkpoint", "field", "deal_index", "difference"])
        for checkpoint in FILE_CHECKPOINTS:
            for deal_index in range(NUM_FILE_DEALS):
                # Each value says where it came from: deal 1 is DEAL_STEP more than deal 0.
                value = checkpoint + DEAL_STEP * deal_index
                writer.writerow([checkpoint, "heuristic", deal_index, value])
                writer.writerow([checkpoint, "greedy", deal_index, OTHER_FIELD_VALUE])

    # Checkpoints 10, 20, 30: deal 0 averages 20, deal 1 averages 120. Greedy rows are -1000,
    # so mixing fields would show.
    assert end_scores(path, "heuristic", NAMED_CHECKPOINTS) == [20.0, 120.0]


def known_band():
    """The 3 control seeds' end scores: shared per-deal noise plus each seed's offset."""
    rng = random.Random(DEV_SEED)
    base = []
    for _ in range(NUM_DEALS):
        base.append(rng.gauss(0.0, DEAL_SPREAD))
    control_scores = []
    for offset in CONTROL_OFFSETS:
        seed_scores = []
        for value in base:
            seed_scores.append(value + offset)
        control_scores.append(seed_scores)
    return base, control_scores


def test_verdict_known_answers():
    base, control_scores = known_band()
    noise_rng = random.Random(DEV_SEED + 1)

    def arm_at(offset):
        # Arm = shared noise + offset + a little noise of its own, so the interval has width.
        scores = []
        for value in base:
            scores.append(value + offset + noise_rng.gauss(0.0, ARM_NOISE))
        return scores

    # catches: a rule that never says candidate. Arm 0.3 clears every seed (best is 0.2).
    label, mean, low, high = arm_verdict(
        arm_at(CLEAR_WIN_OFFSET), control_scores, random.Random(DEV_SEED)
    )
    assert label == CANDIDATE
    assert low > 0

    # catches: a wrong sign in the mirrored test. Arm -0.1 is under every seed (worst is 0).
    label, mean, low, high = arm_verdict(
        arm_at(CLEAR_LOSS_OFFSET), control_scores, random.Random(DEV_SEED)
    )
    assert label == LOSES
    assert high < 0

    # catches: condition (b) dropped. Arm 0.15 beats the band's mean (0.1) on the interval,
    # but not the best seed (0.2): a tie.
    label, mean, low, high = arm_verdict(
        arm_at(BETWEEN_OFFSET), control_scores, random.Random(DEV_SEED)
    )
    assert low > 0
    assert label == TIE
