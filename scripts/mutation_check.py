"""
Mutation check of the DMC loop's and the checkpoint evaluation's gates (stage plan
specialist_explore, stage 4 pieces 2 and 9).

A gate is only worth trusting if it goes red when the code is wrong. This script plants one
known bug at a time in a copy of the code, runs the gate files against the copy, and records
which gates failed. A bug that leaves every gate green is a blind spot: it gets a new gate.
The repo's own files are never edited. The copy has no data/ folder, so no planted bug can
reach the sealed deals: one that asks for them gets a missing-file error.

Some bugs only show on one kind of machine. A batch left on the CPU is harmless when the net is
on the CPU too, so the device bugs are planted only where there is a CUDA GPU (teapot); the
"no GPU" refusal only where there is none (the laptop). A full record needs a run on each.

Run from the repo root (about six minutes):
    PYTHONPATH=src .venv/bin/python scripts/mutation_check.py
Output: a table on screen, and under results/specialist_explore/ the file
mutation_check_dmc.json (machine without a GPU) or mutation_check_dmc_cuda.json (with one).
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from evaluate_agents import REPO_ROOT, git_state

# What a copy needs to run the gates: the package, the tests, the scripts the tests start, and
# pyproject.toml, which tells pytest to import from src/ and tests/.
COPIED_DIRECTORIES = ("src", "tests", "scripts")
COPIED_FILES = ("pyproject.toml",)
# The gates under test, relative to the copy's root.
GATES_FILES = ("tests/test_dmc.py", "tests/test_evaluate_checkpoints.py")
# Longest one gate run may take before it is recorded as hung (a normal run is under 20 s).
GATES_TIMEOUT_SECONDS = 600
RESULTS_DIR = REPO_ROOT / "results" / "specialist_explore"

DMC = "src/stacker/dmc.py"
TRAIN_SCRIPT = "scripts/train_dmc.py"
EVAL_SCRIPT = "scripts/evaluate_checkpoints.py"
# The line of shuffle_returns that takes one sample apart. A planted bug adds a line after it.
SHUFFLE_UNPACK = "        state_vector, history, history_mask, taken_action, _ = samples[index]\n"
# One planted bug each: (name, file, text to find, text to put in its place). The text to find
# must occur exactly once in the file. Where the same line occurs in two functions, the
# leading newline and indentation pick the one meant (16 spaces: inside train's update loop).
# Planted on every machine.
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
    (
        "shuffle moves the actions with the returns",
        DMC,
        SHUFFLE_UNPACK,
        SHUFFLE_UNPACK + "        taken_action = samples[(index + 1) % len(samples)][3]\n",
    ),
    (
        "shuffle leaves the returns in order",
        DMC,
        "    rng.shuffle(shuffled)\n",
        "",
    ),
    (
        "shuffle draws returns with repeats",
        DMC,
        "    rng.shuffle(shuffled)\n",
        "    shuffled = rng.choices(shuffled, k=len(shuffled))\n",
    ),
    (
        "shuffle changes the batch it was handed",
        DMC,
        "    return shuffled_samples\n",
        "    samples[:] = shuffled_samples\n    return shuffled_samples\n",
    ),
    (
        "shuffle flag ignored",
        DMC,
        "                if shuffle_control:\n",
        "                if False:\n",
    ),
    (
        "shuffle result thrown away",
        DMC,
        "batch = shuffle_returns(batch, shuffle_rng)",
        "shuffle_returns(batch, shuffle_rng)",
    ),
    (
        "return variance divides by B - 1",
        DMC,
        "var(correction=0)",
        "var(correction=1)",
    ),
    (
        "samples_total counts updates",
        DMC,
        '"samples_total": updates_done * samples_per_update,',
        '"samples_total": updates_done,',
    ),
    (
        "train starts from another seed",
        DMC,
        "    torch.manual_seed(seed)\n    q_net = QNet()\n\n",
        "    torch.manual_seed(seed + 1)\n    q_net = QNet()\n\n",
    ),
    (
        "checkpoint 0 is not the starting net",
        DMC,
        'torch.save(actor_net.state_dict(), run_dir / f"checkpoint_{0:06d}.pt")',
        'torch.save(QNet().state_dict(), run_dir / f"checkpoint_{0:06d}.pt")',
    ),
    (
        "evaluation makes random moves",
        EVAL_SCRIPT,
        "EVAL_EPSILON = 0.0",
        "EVAL_EPSILON = 0.05",
    ),
    (
        "checkpoint net left in training mode",
        EVAL_SCRIPT,
        "    net.eval()\n",
        "    net.train()\n",
    ),
    (
        "bootstrap rng not named by the seed",
        EVAL_SCRIPT,
        'random.Random(f"{seed}:bootstrap")',
        "random.Random()",
    ),
    (
        "evaluation asks for the sealed deals",
        EVAL_SCRIPT,
        "select_deals(num_players, num_deals, seed, False)",
        "select_deals(num_players, num_deals, seed, True)",
    ),
    (
        "eval script accepts a test seed",
        EVAL_SCRIPT,
        "    if args.seed < DEV_SEED_MIN:\n",
        "    if False:\n",
    ),
)

# Planted only where there is a CUDA GPU: with the net on the CPU these change nothing.
CUDA_MUTATIONS = (
    (
        "net inputs not moved to the device",
        DMC,
        "inputs_on_device.append(tensor.to(device))",
        "inputs_on_device.append(tensor)",
    ),
    (
        "returns not moved to the device",
        DMC,
        "    returns = returns.to(device)\n",
        "",
    ),
    (
        "learner never moved to the device",
        DMC,
        "    q_net.to(device)\n",
        "",
    ),
    (
        "checkpoints saved from the learner's net",
        DMC,
        'actor_net.state_dict(), run_dir / f"checkpoint_{updates_done:06d}.pt"',
        'q_net.state_dict(), run_dir / f"checkpoint_{updates_done:06d}.pt"',
    ),
)

# Planted only where there is no CUDA GPU: with one, --device cuda is valid and its gate is
# skipped.
NO_CUDA_MUTATIONS = (
    (
        "train script accepts cuda with no GPU",
        TRAIN_SCRIPT,
        '    if args.device.startswith("cuda") and not torch.cuda.is_available():\n',
        "    if False:\n",
    ),
)


def main():
    """Runs the gates once unchanged, then once per planted bug; prints and saves the table."""
    commit, uncommitted_changes = git_state()

    # The first run plants nothing. It must be all green, or a red gate later proves nothing.
    runs = [("baseline (no bug)", None, None, None)]
    for mutation in MUTATIONS:
        runs.append(mutation)

    # The bugs that only show on this kind of machine are planted; the others are named in the
    # result as not planted, so a reader sees they were left out, not missed.
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        planted_here = CUDA_MUTATIONS
        left_out = NO_CUDA_MUTATIONS
        results_path = RESULTS_DIR / "mutation_check_dmc_cuda.json"
    else:
        planted_here = NO_CUDA_MUTATIONS
        left_out = CUDA_MUTATIONS
        results_path = RESULTS_DIR / "mutation_check_dmc.json"
    for mutation in planted_here:
        runs.append(mutation)
    not_planted = []
    for mutation in left_out:
        not_planted.append(mutation[0])

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
                sys.executable, "-m", "pytest", *GATES_FILES,
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

    print(f"not planted on this machine: {not_planted}")

    result = {
        "gates_files": list(GATES_FILES),
        "commit": commit,
        "uncommitted_changes": uncommitted_changes,
        "cuda": has_cuda,
        "rows": rows,
        "missed": missed,
        "not_planted": not_planted,
    }
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {results_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
