"""
Gates for scripts/evaluate_checkpoints.py (stage plan specialist_explore, stage 4 piece 9;
spec part 14, equation 15).

Eval hygiene: a loaded checkpoint is in eval mode, on the CPU, with the saved weights; the agent
plays the net's highest-scoring action on every turn that has a choice and never calls the net
on a forced turn; scoring the same checkpoint twice gives the same numbers, and the bootstrap's
rng is named by the seed.
The split: scoring a checkpoint never loads the sealed deals; a test seed is refused by the
function and by the script, before anything is written.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_net import DEV_SEED, seeded_net

from stacker.encode import encode_decision
from stacker.evaluate import bootstrap_ci, paired_differences
from stacker.net import QNet, collate_decisions

# scripts/ is not a package, and evaluate_checkpoints.py imports its neighbours by name
# (evaluate_agents, train_dmc), so the folder goes on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_agents  # noqa: E402
import evaluate_checkpoints  # noqa: E402
from evaluate_agents import AGENTS, select_deals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_PLAYERS = 4
# Few deals: each is played in all 4 rotations, enough for hundreds of turns.
GATE_DEALS = 5
FIELD_NAME = "random"
# A test seed (rule 2: 0-2 are for confirm runs) and the run name the refused run is given.
REFUSED_SEED = 2
REFUSED_RUN_NAME = "gate_refused_test_seed"
# A config.json from before stage 5: none of the net keys, so the control net at window 15.
PRE_STAGE_5_CONFIG = {}
# The widest arm (decision log 2026-11-22), as a full stage 5 config's net keys.
WIDE_CONFIG = {"width": 1024, "hidden_layers": 5, "history_hidden": 128, "history_window": 15}


def save_seeded_checkpoint(tmp_path):
    """Saves seeded_net()'s weights as a checkpoint file under tmp_path. Returns its path."""
    path = tmp_path / "checkpoint_000000.pt"
    torch.save(seeded_net().state_dict(), path)
    return path


def test_loaded_checkpoint_is_ready_to_play(tmp_path):
    net, history_window = evaluate_checkpoints.load_checkpoint_net(
        save_seeded_checkpoint(tmp_path), PRE_STAGE_5_CONFIG
    )
    assert history_window == 15

    # Eval mode: nothing in the net behaves as it does in training.
    assert net.training is False

    saved_weights = seeded_net().state_dict()
    loaded_weights = net.state_dict()
    assert set(loaded_weights) == set(saved_weights)
    for name in saved_weights:
        assert torch.equal(loaded_weights[name], saved_weights[name]), name
        assert loaded_weights[name].device.type == "cpu", name


def test_agent_plays_the_highest_scoring_action(tmp_path):
    net, history_window = evaluate_checkpoints.load_checkpoint_net(
        save_seeded_checkpoint(tmp_path), PRE_STAGE_5_CONFIG
    )
    # Same weights in a second net, so working out the expected action adds no call to `net`.
    reference_net = seeded_net()

    net_calls = []

    def count_call(module, inputs, output):
        net_calls.append(1)

    # A forward hook runs once after every forward pass of the net.
    net.register_forward_hook(count_call)

    forced_turns = []
    choice_turns = []

    def watched_agent(observation, actions, rng):
        calls_before = len(net_calls)
        action = evaluate_checkpoints.checkpoint_agent(
            net, history_window, observation, actions, rng
        )

        if len(actions) == 1:
            # Forced turn: the only action, and the net was not called.
            forced_turns.append(1)
            assert action == actions[0]
            assert len(net_calls) == calls_before
            return action

        # A turn with a choice: one net call, and the action is the one the net scores highest.
        # This holds on every such turn, so the agent made no random move (epsilon is 0).
        choice_turns.append(1)
        assert len(net_calls) == calls_before + 1
        batch = collate_decisions([encode_decision(observation, actions)])
        with torch.no_grad():
            q_values = reference_net(*batch)  # [1, K]
        assert action == actions[int(q_values[0].argmax())]
        return action

    deck_orders = select_deals(NUM_PLAYERS, GATE_DEALS, DEV_SEED, False)
    paired_differences(
        deck_orders,
        NUM_PLAYERS,
        watched_agent,
        AGENTS[FIELD_NAME],
        evaluate_checkpoints.TEST_SEATS,
        DEV_SEED,
    )

    # Both kinds of turn came up, so neither branch above passed by never running.
    assert len(forced_turns) > 0
    assert len(choice_turns) > 0


def test_same_checkpoint_scores_the_same_twice(tmp_path, monkeypatch):
    # Record the first number of the rng the bootstrap is handed, then run the real bootstrap.
    # With this few deals the interval itself can come out equal from two different rngs.
    bootstrap_draws = []

    def recording_bootstrap(differences, rng):
        bootstrap_draws.append(rng.random())
        return bootstrap_ci(differences, rng)

    monkeypatch.setattr(evaluate_checkpoints, "bootstrap_ci", recording_bootstrap)

    task = (
        str(save_seeded_checkpoint(tmp_path)),
        FIELD_NAME,
        NUM_PLAYERS,
        GATE_DEALS,
        DEV_SEED,
        PRE_STAGE_5_CONFIG,
        None,
    )
    first = evaluate_checkpoints.evaluate_checkpoint(task)
    second = evaluate_checkpoints.evaluate_checkpoint(task)

    # seconds is play time on the clock; everything else comes from the weights and the seed.
    del first["seconds"]
    del second["seconds"]
    assert first == second
    assert len(first["differences"]) == GATE_DEALS

    # The bootstrap's rng is named by the seed alone, as in evaluate_agents.py.
    expected_draw = random.Random(f"{DEV_SEED}:bootstrap").random()
    assert bootstrap_draws == [expected_draw, expected_draw]


def test_a_1024_wide_checkpoint_loads_and_scores(tmp_path):
    # catches: a loader that builds the control net whatever the config says.
    torch.manual_seed(DEV_SEED)
    path = tmp_path / "checkpoint_000000.pt"
    torch.save(QNet(width=1024).state_dict(), path)
    task = (str(path), FIELD_NAME, NUM_PLAYERS, GATE_DEALS, DEV_SEED, WIDE_CONFIG, None)
    result = evaluate_checkpoints.evaluate_checkpoint(task)
    assert len(result["differences"]) == GATE_DEALS


def test_a_config_missing_one_net_key_is_refused(tmp_path):
    # catches: a gap in a stage 5 config filled in with the control's value.
    partial_config = dict(WIDE_CONFIG)
    del partial_config["history_window"]
    with pytest.raises(ValueError, match="history_window"):
        evaluate_checkpoints.load_checkpoint_net(save_seeded_checkpoint(tmp_path), partial_config)


def test_decisions_are_encoded_at_the_runs_window(tmp_path, monkeypatch):
    # catches: eval encoding at 15 whatever window the run trained with.
    history_rows_seen = []

    def recording_choose_action(net, decision, epsilon, rng):
        history_rows_seen.append(decision[1].shape[0])
        return 0

    monkeypatch.setattr(evaluate_checkpoints, "choose_action", recording_choose_action)
    window_zero_config = dict(evaluate_checkpoints.CONTROL_NET_CONFIG, history_window=0)
    task = (
        str(save_seeded_checkpoint(tmp_path)),
        FIELD_NAME,
        NUM_PLAYERS,
        GATE_DEALS,
        DEV_SEED,
        window_zero_config,
        None,
    )
    evaluate_checkpoints.evaluate_checkpoint(task)
    assert len(history_rows_seen) > 0
    assert set(history_rows_seen) == {0}


def test_field_net_is_built_from_its_own_config(tmp_path):
    # catches: the field net built from the test run's config (a 1024-wide file in a 512 net).
    torch.manual_seed(DEV_SEED)
    wide_path = tmp_path / "checkpoint_wide.pt"
    torch.save(QNet(width=1024).state_dict(), wide_path)
    task = (
        str(save_seeded_checkpoint(tmp_path)),
        "net",
        NUM_PLAYERS,
        GATE_DEALS,
        DEV_SEED,
        PRE_STAGE_5_CONFIG,
        (str(wide_path), WIDE_CONFIG),
    )
    result = evaluate_checkpoints.evaluate_checkpoint(task)
    assert len(result["differences"]) == GATE_DEALS


def test_script_scores_a_net_against_itself_as_exactly_zero(tmp_path):
    # catches: --field-run parsed and then ignored (the field stays an agent, so not 0), or the
    # result written over the run's curve against agents.
    # Identical deterministic nets in every seat play the same game in every rotation, so the
    # test seat covers every seat of one game and each paired difference is exactly 0.
    run_name = "eval-gate-self-play"
    run_dir = evaluate_checkpoints.RUNS_DIR / run_name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    torch.save(seeded_net().state_dict(), run_dir / "checkpoint_000000.pt")
    run_config = {
        "players": NUM_PLAYERS,
        "samples_per_update": 1,
        "git_commit": "gate",
        "uncommitted_changes": False,
    }
    with open(run_dir / "config.json", "w") as file:
        json.dump(run_config, file)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    finished = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "evaluate_checkpoints.py"),
            "--run", run_name, "--field-run", run_name, "--field-checkpoint", "0",
            "--deals", str(GATE_DEALS), "--seed", str(DEV_SEED), "--workers", "1",
        ],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
    )

    # Read what the asserts need, then remove the run and its results.
    result_paths = sorted(evaluate_checkpoints.RESULTS_DIR.glob(f"{run_name}_*"))
    differences = []
    for path in result_paths:
        if path.name.endswith("_deals.csv"):
            with open(path) as file:
                for row in csv.DictReader(file):
                    differences.append(float(row["difference"]))
    result_names = [path.name for path in result_paths]
    for path in result_paths:
        path.unlink()
    shutil.rmtree(run_dir)

    assert finished.returncode == 0, finished.stderr
    for name in result_names:
        assert f"_vs_{run_name}_ckpt0" in name
    assert len(differences) == GATE_DEALS
    assert differences == [0.0] * GATE_DEALS


def test_scoring_never_loads_the_sealed_deals(tmp_path, monkeypatch):
    def refuse_sealed(num_players):
        raise AssertionError("evaluate_checkpoint asked for the sealed deals")

    # select_deals is the only caller of load_sealed, and looks the name up in its own module,
    # so this stand-in runs if any path reaches the sealed set.
    monkeypatch.setattr(evaluate_agents, "load_sealed", refuse_sealed)

    task = (
        str(save_seeded_checkpoint(tmp_path)),
        FIELD_NAME,
        NUM_PLAYERS,
        GATE_DEALS,
        DEV_SEED,
        PRE_STAGE_5_CONFIG,
        None,
    )
    result = evaluate_checkpoints.evaluate_checkpoint(task)
    assert len(result["differences"]) == GATE_DEALS


def test_function_refuses_a_test_seed(tmp_path):
    # The real load_sealed is in place here: a wrong path to the sealed deals would return them
    # and not raise.
    task = (
        str(save_seeded_checkpoint(tmp_path)),
        FIELD_NAME,
        NUM_PLAYERS,
        GATE_DEALS,
        REFUSED_SEED,
        PRE_STAGE_5_CONFIG,
        None,
    )
    with pytest.raises(ValueError, match="dev runs use seeds"):
        evaluate_checkpoints.evaluate_checkpoint(task)


def test_script_refuses_a_test_seed():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate_checkpoints.py"),
        "--run", REFUSED_RUN_NAME, "--seed", str(REFUSED_SEED),
    ]
    finished = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
    )
    assert finished.returncode != 0
    # The seed is what stopped it. The run does not exist either, and that would also exit
    # with an error, so the exit code alone proves nothing.
    assert "dev seeds are" in finished.stderr
    # The refusal comes before anything is written.
    assert list(evaluate_checkpoints.RESULTS_DIR.glob(REFUSED_RUN_NAME + "*")) == []


def test_mixed_table_seats_its_named_agents_and_the_net_only_where_listed():
    # catches: a table seat wired to the wrong kind (the net seat given an agent, or an agent
    # seat given the net), seats out of table order, or a net silently dropped from a no-net table.
    net_stand_in = "the field net"
    styles = evaluate_checkpoints.mixed_table_field("styles", net_stand_in)
    assert styles == [AGENTS["heuristic"], AGENTS["greedy"], net_stand_in]
    weak = evaluate_checkpoints.mixed_table_field("weak", None)
    assert weak == [AGENTS["random"], AGENTS["greedy"], AGENTS["heuristic"]]
    with pytest.raises(ValueError):
        evaluate_checkpoints.mixed_table_field("weak", net_stand_in)
    with pytest.raises(ValueError):
        evaluate_checkpoints.mixed_table_field("styles", None)


def test_each_mixed_table_writes_its_own_files(tmp_path):
    # catches: two tables written to one file name (the second replaces the first), or the
    # table missing from the field column the plots read.
    run_name = "eval-gate-mixed-tables"
    run_dir = evaluate_checkpoints.RUNS_DIR / run_name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    torch.save(seeded_net().state_dict(), run_dir / "checkpoint_000000.pt")
    run_config = {
        "players": NUM_PLAYERS,
        "samples_per_update": 1,
        "git_commit": "gate",
        "uncommitted_changes": False,
    }
    with open(run_dir / "config.json", "w") as file:
        json.dump(run_config, file)

    # The weak table has no net seat; the styles table seats this run's own net in it.
    table_arguments = [
        ["--mixed-table", "weak"],
        ["--mixed-table", "styles", "--field-run", run_name, "--field-checkpoint", "0"],
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return_codes = []
    for arguments in table_arguments:
        finished = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "evaluate_checkpoints.py"),
                "--run", run_name, "--deals", str(GATE_DEALS), "--seed", str(DEV_SEED),
                "--workers", "1",
            ] + arguments,
            cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
        )
        return_codes.append((finished.returncode, finished.stderr))

    # Read what the asserts need, then remove the run and its results.
    result_paths = sorted(evaluate_checkpoints.RESULTS_DIR.glob(f"{run_name}_*"))
    fields_by_name = {}
    for path in result_paths:
        if path.name.endswith(".csv"):
            with open(path) as file:
                for row in csv.DictReader(file):
                    fields_by_name[path.name] = row["field"]
    for path in result_paths:
        path.unlink()
    shutil.rmtree(run_dir)

    for return_code, stderr in return_codes:
        assert return_code == 0, stderr
    # Three files per table: curve, summary, per-deal.
    assert len(result_paths) == 6
    weak_curve = f"{run_name}_eval_dev_n{GATE_DEALS}_seed{DEV_SEED}_mixed-weak.csv"
    styles_curve = (
        f"{run_name}_eval_dev_n{GATE_DEALS}_seed{DEV_SEED}_mixed-styles_vs_{run_name}_ckpt0.csv"
    )
    assert fields_by_name[weak_curve] == "mixed-weak"
    assert fields_by_name[styles_curve] == "mixed-styles"
