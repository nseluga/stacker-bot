"""
Mutation check of the DMC loop's, the checkpoint evaluation's and the stage 5 arms' gates
(stage plan specialist_explore, stage 4 pieces 2 and 9; stage 5 piece 9b: net sizes, run
settings, net against net, the arm rule and the arm plots; actor-throughput piece 1b: the
round-trip probe; the fast actor's build, piece P7: batched picks, lockstep hands, the fast
actor process, the fast path in train and its flags).

A gate is only worth trusting if it goes red when the code is wrong. This script plants one
known bug at a time in a copy of the code, runs the gate files against the copy, and records
which gates failed. A bug that leaves every gate green is a blind spot: it gets a new gate.
The repo's own files are never edited. The copy has no data/ folder, so no planted bug can
reach the sealed deals: one that asks for them gets a missing-file error.

Some bugs only show on one kind of machine. A batch left on the CPU is harmless when the net is
on the CPU too, so the device bugs are planted only where there is a CUDA GPU (teapot); the
"no GPU" refusal only where there is none (the laptop). A full record needs a run on each.

Run from the repo root (60-100 min in full: each of ~60 runs takes 1-2 min):
    PYTHONPATH=src .venv/bin/python scripts/mutation_check.py
Two flags make it shorter. --first-red stops each gate run at its first red gate, so a row names
one failed gate, not all of them; that still settles red or green. --resume-log PATH reads the
screen output of a run that died part way and copies its finished bugs into the result instead
of planting them again; the baseline always runs again, so the "no bug is green" check is fresh.
    PYTHONPATH=src .venv/bin/python scripts/mutation_check.py --first-red \
        --resume-log scratch/mutation_fast_actor_laptop_crashed.log
Output: a table on screen, and under results/specialist_explore/ the file
mutation_check_fast_actor.json (machine without a GPU) or mutation_check_fast_actor_cuda.json
(with one). The earlier records, mutation_check_dmc*.json (stage 4) and mutation_check_arms*.json
(stage 5 before the fast actor), are from their own commits and are not overwritten.
"""

import argparse
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
# Makes the copy a git repo with one commit; the name and email are set here so no global git
# config is needed, and they name no person.
COPY_GIT_COMMANDS = (
    ("git", "init", "-q"),
    ("git", "add", "-A"),
    (
        "git", "-c", "user.name=mutation-check", "-c", "user.email=mutation-check@localhost",
        "commit", "-q", "-m", "unchanged copy",
    ),
)
# The gates under test, relative to the copy's root.
GATES_FILES = (
    "tests/test_dmc.py",
    "tests/test_evaluate_checkpoints.py",
    "tests/test_net.py",
    "tests/test_encode.py",
    "tests/test_arms.py",
    "tests/test_plot_arms.py",
    "tests/test_round_trip_probe.py",
)
# The screen line for a bug that turned no gate red; resumed rows need it to read as green.
NO_GATE_FAILED = "no gate failed"
# Separates the failed gates on one screen line (gate names never contain it).
GATES_SEPARATOR = ", "
# Longest one gate run may take before it is recorded as hung (a normal run is about 40 s).
GATES_TIMEOUT_SECONDS = 600
RESULTS_DIR = REPO_ROOT / "results" / "specialist_explore"
# The result file's name without its ending; "_cuda" is added on a machine with a GPU.
RESULTS_STEM = "mutation_check_fast_actor"

DMC = "src/stacker/dmc.py"
NET = "src/stacker/net.py"
ARMS = "src/stacker/arms.py"
PLOT_ARMS_SCRIPT = "scripts/plot_arms.py"
TRAIN_SCRIPT = "scripts/train_dmc.py"
ROUND_TRIP_SCRIPT = "scripts/round_trip_probe.py"
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
        # Four spaces of indent: choose_action's line, not the batched pick's.
        "\n    if num_actions == 1:",
        "\n    if num_actions == 0:",
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
        # Twenty spaces: the old path's copy after each update, not the fast path's checkpoint.
        "\n                    actor_net.load_state_dict(q_net.state_dict())\n",
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
        "    torch.manual_seed(seed)\n    q_net = QNet(width=width,",
        "    torch.manual_seed(seed + 1)\n    q_net = QNet(width=width,",
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
    # Stage 5 (piece 9b): one bug per new code path of pieces 1-8.
    (
        "width argument ignored",
        NET,
        "        super().__init__()\n",
        "        super().__init__()\n        width = DEFAULT_MLP_WIDTH\n",
    ),
    (
        "depth argument ignored",
        NET,
        "for _ in range(hidden_layers):",
        "for _ in range(DEFAULT_HIDDEN_LAYERS):",
    ),
    (
        "summary size not passed to the MLP's input",
        NET,
        "layer_input_dim = history_hidden + STATE_DIM + ACTION_DIM",
        "layer_input_dim = DEFAULT_HISTORY_HIDDEN + STATE_DIM + ACTION_DIM",
    ),
    (
        "actors encode at the fixed window, not the run's",
        DMC,
        "hand_rng, history_window, chair_seat",
        "hand_rng, HISTORY_WINDOW, chair_seat",
    ),
    (
        "optimizer name ignored",
        DMC,
        '    if optimizer == "rmsprop":\n',
        "    if True:\n",
    ),
    (
        "net plays the chair seat",
        DMC,
        "        if seat == chair_seat:\n",
        "        if False:\n",
    ),
    (
        "chair seat draw never picks the last seat",
        DMC,
        "    return hand_rng.randrange(num_seats)\n",
        "    return hand_rng.randrange(num_seats - 1)\n",
    ),
    (
        "actor ignores the heuristic chair",
        DMC,
        "        if heuristic_chair:\n",
        "        if False:\n",
    ),
    (
        "train drops the heuristic chair before the actors",
        DMC,
        "                    heuristic_chair,\n",
        "                    False,\n",
    ),
    (
        "script drops the heuristic chair key",
        TRAIN_SCRIPT,
        '        config["heuristic_chair"] = True\n',
        '        config["heuristic_chair"] = False\n',
    ),
    (
        "learning rate ignored",
        DMC,
        "            lr=learning_rate,\n            alpha=RMSPROP_ALPHA,",
        "            lr=LEARNING_RATE,\n            alpha=RMSPROP_ALPHA,",
    ),
    (
        "arm's settings not checked against its base run",
        TRAIN_SCRIPT,
        "        if config.get(key) != base_config.get(key):\n",
        "        if False:\n",
    ),
    (
        "checkpoint net built with the control sizes, not its run's",
        EVAL_SCRIPT,
        '    net = QNet(\n        width=net_config["width"],\n'
        '        hidden_layers=net_config["hidden_layers"],\n'
        '        history_hidden=net_config["history_hidden"],\n    )',
        "    net = QNet()",
    ),
    (
        "field net built from the tested run's config",
        EVAL_SCRIPT,
        "load_checkpoint_net(field_path, field_config)",
        "load_checkpoint_net(field_path, run_config)",
    ),
    (
        "arm rule drops 'above every control seed'",
        ARMS,
        "if ci_low > 0 and arm_end > max(control_ends):",
        "if ci_low > 0:",
    ),
    (
        "arm rule's loses test has the wrong sign",
        ARMS,
        "elif ci_high < 0 and arm_end < min(control_ends):",
        "elif ci_high > 0 and arm_end < min(control_ends):",
    ),
    (
        "long control seed read at the end of its file, not at arm length",
        PLOT_ARMS_SCRIPT,
        '        if row["checkpoint"] <= arm_length:\n            checkpoints.append',
        "        if True:\n            checkpoints.append",
    ),
    (
        "round-trip server replies to actor 0 always",
        ROUND_TRIP_SCRIPT,
        "reply_queues[row_actors[row]].put([(row_slots[row], row_ids[row], answers[row])])",
        "reply_queues[0].put([(row_slots[row], row_ids[row], answers[row])])",
    ),
    (
        "round-trip server reverses the answers in a batch",
        ROUND_TRIP_SCRIPT,
        "reply_queues[row_actors[row]].put([(row_slots[row], row_ids[row], answers[row])])",
        "reply_queues[row_actors[row]].put("
        "[(row_slots[row], row_ids[row], answers[len(decisions) - 1 - row])])",
    ),
    (
        "bulk server replies to actor 0 always",
        ROUND_TRIP_SCRIPT,
        "reply_queues[actor_id].put(replies_by_actor[actor_id])",
        "reply_queues[0].put(replies_by_actor[actor_id])",
    ),
    (
        "bulk server pairs answers with the wrong requests",
        ROUND_TRIP_SCRIPT,
        "replies_by_actor[actor_id].append((row_slots[row], row_ids[row], answers[row]))",
        "replies_by_actor[actor_id].append("
        "(row_slots[row], row_ids[row], answers[len(decisions) - 1 - row]))",
    ),
    (
        "every actor routed to server 0",
        ROUND_TRIP_SCRIPT,
        "request_queues[actor_id % num_servers],",
        "request_queues[0],",
    ),
    (
        "actor net pairs answers with the wrong rows",
        ROUND_TRIP_SCRIPT,
        "batch_answers = batched_choices(net, batch, device)",
        "batch_answers = batched_choices(net, batch, device)[::-1]",
    ),
    (
        "learner never writes the second GPU's shared net",
        DMC,
        "    for shared_net in shared_nets:\n        copy_start = time.perf_counter()",
        "    for shared_net in shared_nets[:1]:\n        copy_start = time.perf_counter()",
    ),
    (
        "actor never refreshes its own net",
        DMC,
        "        private_net.load_state_dict(shared_net.state_dict())\n",
        "        shared_net.state_dict()\n",
    ),
    (
        "refresh keeps a copy a write overlapped",
        DMC,
        "        if version.value == before:\n",
        "        if True:\n",
    ),
    (
        "run speedup cap not shrunk by the learner's slowdown",
        ROUND_TRIP_SCRIPT,
        "cap = learner_cap / learner_slowdown",
        "cap = learner_cap",
    ),
)

# The fast actor's build (P7): one bug per gate it added.
FAST_ACTOR_MUTATIONS = (
    (
        "batched pick loses the padding mask",
        DMC,
        '    masked_q = q_values.masked_fill(~action_mask, float("-inf"))\n',
        "    masked_q = q_values\n",
    ),
    (
        "batched pick draws on a forced turn",
        DMC,
        "        if num_actions == 1:\n            choices.append(0)\n",
        "        if num_actions == 1:\n            rng.random()\n            choices.append(0)\n",
    ),
    (
        "batched greedy answers in reverse order",
        DMC,
        "            choices[greedy_positions[index]] = greedy_choices[index]",
        "            choices[greedy_positions[index]] = greedy_choices[-1 - index]",
    ),
    (
        "lockstep records all land in the first slot",
        DMC,
        "        slot.records.append(",
        "        playing_slots[0].records.append(",
    ),
    (
        "fast actor reuses a hand index on refill",
        DMC,
        "                slots[position] = deal_slot(seed, actor_id, next_hand_index)\n"
        "                next_hand_index += 1\n",
        "                slots[position] = deal_slot(seed, actor_id, next_hand_index)\n",
    ),
    (
        "fast actor's stream leaves out the actor id",
        DMC,
        "    rng = random.Random(hand_stream_name(seed, actor_id, hand_index))\n"
        "    return HandSlot",
        "    rng = random.Random(hand_stream_name(seed, 0, hand_index))\n    return HandSlot",
    ),
    (
        "fast actor never refreshes",
        DMC,
        "        held_updates, _ = refresh_private_net(\n"
        "            own_net, shared_net, version, held_updates, FAST_ACTOR_REFRESH_EVERY\n"
        "        )\n",
        "        held_updates = 0\n",
    ),
    (
        "fast path never writes the shared nets",
        DMC,
        "                    write_weights(q_net, shared_nets, version)\n",
        "                    pass\n",
    ),
    (
        "fast checkpoint misses the learner's latest weights",
        DMC,
        "                    if fast_actor:\n"
        "                        actor_net.load_state_dict(q_net.state_dict())\n",
        "",
    ),
    (
        "plain run writes fast_actor false",
        TRAIN_SCRIPT,
        "        return {}, set()\n",
        '        return {"fast_actor": False}, set()\n',
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
    (
        "fast actor's net left where its copy was made",
        DMC,
        "    own_net = copy.deepcopy(shared_net).to(device)\n",
        "    own_net = copy.deepcopy(shared_net)\n",
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


def rows_from_log(log_text, names):
    """
    Reads the finished bugs out of an earlier run's screen output.
    log_text: str, what main printed, one "<bug name>: <failed gates>" line per finished bug.
    names: the bug names this run would plant.
    Returns a dict from bug name to its list of failed gates, for every name with a line.
    A line is matched by its known name, not by splitting at ":", since some names hold commas.
    """
    finished = {}
    for line in log_text.splitlines():
        for name in names:
            prefix = name + ": "
            if not line.startswith(prefix):
                continue
            verdict = line[len(prefix):]
            if verdict == NO_GATE_FAILED:
                finished[name] = []
            else:
                finished[name] = verdict.split(GATES_SEPARATOR)
    return finished


def main():
    """Runs the gates once unchanged, then once per planted bug; prints and saves the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-red", action="store_true",
                        help="stop each gate run at its first red gate (pytest -x)")
    parser.add_argument("--resume-log", default=None,
                        help="screen output of an earlier run whose finished bugs are copied")
    args = parser.parse_args()
    commit, uncommitted_changes = git_state()

    # The first run plants nothing. It must be all green, or a red gate later proves nothing.
    runs = [("baseline (no bug)", None, None, None)]
    for mutation in MUTATIONS:
        runs.append(mutation)
    for mutation in FAST_ACTOR_MUTATIONS:
        runs.append(mutation)

    # The bugs that only show on this kind of machine are planted; the others are named in the
    # result as not planted, so a reader sees they were left out, not missed.
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        planted_here = CUDA_MUTATIONS
        left_out = NO_CUDA_MUTATIONS
        results_path = RESULTS_DIR / f"{RESULTS_STEM}_cuda.json"
    else:
        planted_here = NO_CUDA_MUTATIONS
        left_out = CUDA_MUTATIONS
        results_path = RESULTS_DIR / f"{RESULTS_STEM}.json"
    for mutation in planted_here:
        runs.append(mutation)
    not_planted = []
    for mutation in left_out:
        not_planted.append(mutation[0])

    # Bugs already finished in the earlier run are copied, not planted again. The baseline is
    # never copied: the green check must come from this run's code.
    resumed = {}
    if args.resume_log is not None:
        bug_names = []
        for mutation in runs[1:]:
            bug_names.append(mutation[0])
        resumed = rows_from_log(Path(args.resume_log).read_text(), bug_names)

    pytest_flags = ["-q", "--tb=no", "-rfE", "-p", "no:cacheprovider"]
    if args.first_red:
        pytest_flags.append("-x")

    rows = []
    for name, relative_path, old_text, new_text in runs:
        if name in resumed:
            rows.append({"mutation": name, "failed_gates": resumed[name],
                         "from_log": args.resume_log})
            # Same line format as a fresh run, so this output can itself be resumed from.
            if len(resumed[name]) == 0:
                print(f"{name}: {NO_GATE_FAILED}", flush=True)
            else:
                print(f"{name}: {GATES_SEPARATOR.join(resumed[name])}", flush=True)
            continue
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

            # The scripts the gates start record the git commit they ran at and refuse to run
            # outside a repo, so the copy becomes a repo with one commit of the unchanged code.
            # The planted bug lands after it, as an uncommitted change.
            for git_command in COPY_GIT_COMMANDS:
                subprocess.run(git_command, cwd=copy_root, capture_output=True, check=True)

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
            # -x (with --first-red): stop at the first red gate.
            command = [sys.executable, "-m", "pytest", *GATES_FILES, *pytest_flags]
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
            verdict = NO_GATE_FAILED
        else:
            verdict = GATES_SEPARATOR.join(failed_gates)
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
        "first_red_only": args.first_red,
        "resume_log": args.resume_log,
        "rows": rows,
        "missed": missed,
        "not_planted": not_planted,
    }
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {results_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
