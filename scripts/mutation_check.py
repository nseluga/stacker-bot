"""
Mutation check of the DMC loop's gates (stage plan specialist_explore, stage 4 piece 2).

A gate is only worth trusting if it goes red when the code is wrong. This script plants one
known bug at a time in a copy of the code, runs tests/test_dmc.py against the copy, and records
which gates failed. A bug that leaves every gate green is a blind spot: it gets a new gate.
The repo's own files are never edited.

Run from the repo root (about a minute):
    PYTHONPATH=src .venv/bin/python scripts/mutation_check.py
Output: a table on screen, and results/specialist_explore/mutation_check_dmc.json.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate_agents import REPO_ROOT, git_state

# What a copy needs to run the gates: the package, the tests, the scripts the tests start, and
# pyproject.toml, which tells pytest to import from src/ and tests/.
COPIED_DIRECTORIES = ("src", "tests", "scripts")
COPIED_FILES = ("pyproject.toml",)
# The gates under test, relative to the copy's root.
GATES_FILE = "tests/test_dmc.py"
# Longest one gate run may take before it is recorded as hung (a normal run is under 20 s).
GATES_TIMEOUT_SECONDS = 600
RESULTS_PATH = REPO_ROOT / "results" / "specialist_explore" / "mutation_check_dmc.json"

DMC = "src/stacker/dmc.py"
# One planted bug each: (name, file, text to find, text to put in its place). The text to find
# must occur exactly once in the file. Where the same line occurs in two functions, the
# leading newline and indentation pick the one meant (16 spaces: inside train's update loop).
MUTATIONS = (
    (
        "return of the next seat",
        DMC,
        "hand_return = float(hand_points[seat])",
        "hand_return = float(hand_points[(seat + 1) % len(hand_points)])",
    ),
    (
        "stored action is row 0, not the one played",
        DMC,
        "taken_action = action_matrix[index : index + 1].copy()",
        "taken_action = action_matrix[0:1].copy()",
    ),
    (
        "forced turn calls the net",
        DMC,
        "if num_actions == 1:",
        "if num_actions == 0:",
    ),
    (
        "greedy pick takes the lowest Q",
        DMC,
        "return int(q_values[0].argmax())",
        "return int(q_values[0].argmin())",
    ),
    (
        "returns shaped [B, 1]",
        DMC,
        "returns = torch.tensor(hand_returns, dtype=torch.float32)",
        "returns = torch.tensor(hand_returns, dtype=torch.float32).unsqueeze(1)",
    ),
    (
        "no gradient clip",
        DMC,
        "clip_grad_norm_(q_net.parameters(), MAX_GRAD_NORM)",
        'clip_grad_norm_(q_net.parameters(), float("inf"))',
    ),
    (
        "old gradients not cleared",
        DMC,
        "    optimizer.zero_grad()\n",
        "",
    ),
    (
        "one-process loop reuses its samples",
        DMC,
        "\n        waiting_samples = waiting_samples[samples_per_update:]\n",
        "\n",
    ),
    (
        "actor loop reuses its samples",
        DMC,
        "\n                waiting_samples = waiting_samples[samples_per_update:]\n",
        "\n",
    ),
    (
        "weights never copied to the actors",
        DMC,
        "\n                actor_net.load_state_dict(q_net.state_dict())\n",
        "\n",
    ),
    (
        "every actor plays the same stream",
        DMC,
        "hand_stream_name(seed, actor_id, hands_played)",
        "hand_stream_name(seed, 0, hands_played)",
    ),
)


def main():
    """Runs the gates once unchanged, then once per planted bug; prints and saves the table."""
    commit, uncommitted_changes = git_state()

    # The first run plants nothing. It must be all green, or a red gate later proves nothing.
    runs = [("baseline (no bug)", None, None, None)]
    for mutation in MUTATIONS:
        runs.append(mutation)

    rows = []
    for name, relative_path, old_text, new_text in runs:
        with tempfile.TemporaryDirectory() as copy_root:
            copy_root = Path(copy_root)
            for directory in COPIED_DIRECTORIES:
                shutil.copytree(
                    REPO_ROOT / directory,
                    copy_root / directory,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            for file_name in COPIED_FILES:
                shutil.copy(REPO_ROOT / file_name, copy_root / file_name)

            if relative_path is not None:
                target = copy_root / relative_path
                source = target.read_text()
                # A bug that was not planted would leave the gates green and read as a blind
                # spot, so a text that is missing, or matches twice, stops the script.
                if source.count(old_text) != 1:
                    raise SystemExit(
                        f"{name}: text found {source.count(old_text)} times in {relative_path}"
                    )
                target.write_text(source.replace(old_text, new_text))

            # -rfE: one "FAILED <test>" or "ERROR <test>" line per red gate. -p no:cacheprovider:
            # no .pytest_cache in the copy. The copy's pyproject.toml points pytest at the
            # copy's src/ and tests/, so the repo's code is not what runs.
            command = [
                sys.executable, "-m", "pytest", GATES_FILE,
                "-q", "--tb=no", "-rfE", "-p", "no:cacheprovider",
            ]
            hung = False
            try:
                finished = subprocess.run(
                    command,
                    cwd=copy_root,
                    capture_output=True,
                    text=True,
                    timeout=GATES_TIMEOUT_SECONDS,
                )
                output = finished.stdout
            except subprocess.TimeoutExpired:
                hung = True
                output = ""

        failed_gates = []
        for line in output.splitlines():
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                # "FAILED tests/test_dmc.py::test_name - reason" -> "test_name"
                test_id = line.split()[1]
                failed_gates.append(test_id.split("::")[-1])
        if hung:
            failed_gates.append(f"(hung: no result in {GATES_TIMEOUT_SECONDS} s)")

        rows.append({"mutation": name, "failed_gates": failed_gates})
        if len(failed_gates) == 0:
            verdict = "no gate failed"
        else:
            verdict = ", ".join(failed_gates)
        print(f"{name}: {verdict}", flush=True)

    if len(rows[0]["failed_gates"]) != 0:
        raise SystemExit("the gates fail with no bug planted: fix that before reading the table")

    missed = []
    for row in rows[1:]:
        if len(row["failed_gates"]) == 0:
            missed.append(row["mutation"])
    print(f"\n{len(rows) - 1} bugs planted, {len(missed)} missed by every gate: {missed}")

    result = {
        "gates_file": GATES_FILE,
        "commit": commit,
        "uncommitted_changes": uncommitted_changes,
        "rows": rows,
        "missed": missed,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
