"""
Gate for scripts/mutation_check.py's resume (fast actor's build, mutation-check speed-up): a log
from a run that died part way gives back exactly the bugs it finished, with their gates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mutation_check import rows_from_log

# Two finished bugs (one red, one green) and one the run never reached. The red name holds a
# comma, as real names do, and the baseline line must not be read as a bug.
RED_NAME = "stored action is row 0, not the one played"
GREEN_NAME = "a bug no gate sees"
UNREACHED_NAME = "never reached"
LOG = (
    "baseline (no bug): no gate failed\n"
    f"{RED_NAME}: test_lockstep_hands_equal_self_play_hand, test_fast_actor_plays_its_own_streams\n"
    f"{GREEN_NAME}: no gate failed\n"
)


def test_log_gives_back_the_finished_bugs_only():
    # catches: an unreached bug read as finished (it would never be planted), a finished bug
    # lost (it would be planted twice), or a red bug's gates split wrong at its comma.
    finished = rows_from_log(LOG, [RED_NAME, GREEN_NAME, UNREACHED_NAME])
    assert finished == {
        RED_NAME: ["test_lockstep_hands_equal_self_play_hand",
                   "test_fast_actor_plays_its_own_streams"],
        GREEN_NAME: [],
    }
