"""
Gates for scripts/same_curve_check.py (stage plan specialist_explore, stage 5 piece 15): a mean
outside the reference interval is a miss and one on its edge is not; curves with no shared
point match nothing; a shared checkpoint with different sample counts is an error, not a match.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from same_curve_check import points_outside

# A reference interval [0.4, 0.6] at checkpoint 125, and how far past its top edge the
# outside mean sits (small, so a sign or comparison slip shows).
CI_LOW = 0.4
CI_HIGH = 0.6
STEP_PAST_EDGE = 0.001
SAMPLES_AT_125 = 400_000


def row(checkpoint, field, mean, samples=SAMPLES_AT_125):
    """One curve row with the reference interval."""
    return {"checkpoint": checkpoint, "samples": samples, "field": field, "mean": mean,
            "ci_low": CI_LOW, "ci_high": CI_HIGH}


def test_edge_is_inside_and_just_past_is_a_miss():
    # catches: a mean just past ci_high passing, or a mean exactly on an edge failing.
    reference = [row(125, "random", 0.5), row(125, "greedy", 0.5)]
    new = [row(125, "random", CI_HIGH), row(125, "greedy", CI_HIGH + STEP_PAST_EDGE)]
    matched, misses = points_outside(reference, new)
    assert matched == 2
    assert len(misses) == 1
    assert misses[0]["field"] == "greedy"
    assert misses[0]["distance"] == pytest.approx(STEP_PAST_EDGE)


def test_no_shared_point_matches_nothing():
    # catches: curves with no shared checkpoint reading as a pass (main fails on matched == 0).
    reference = [row(125, "random", 0.5)]
    new = [row(250, "random", 0.5)]
    matched, misses = points_outside(reference, new)
    assert matched == 0
    assert misses == []


def test_same_checkpoint_different_samples_is_an_error():
    # catches: pairing checkpoints with the same update number but different training amounts.
    reference = [row(125, "random", 0.5)]
    new = [row(125, "random", 0.5, samples=2 * SAMPLES_AT_125)]
    with pytest.raises(ValueError):
        points_outside(reference, new)
