"""
Checks scripts/evaluate_agents.py git_state (stage plan characterization, piece 0): only changes
to code paths set the uncommitted-changes flag; a new result file does not.
"""

import importlib.util
import subprocess
from pathlib import Path

# scripts/ is not a package, so load evaluate_agents.py from its file path.
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_agents.py"
spec = importlib.util.spec_from_file_location("evaluate_agents", SCRIPT_PATH)
evaluate_agents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_agents)


def run_git(repo, *arguments):
    """Runs one git command in repo and fails the test if git fails."""
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def test_only_code_changes_set_the_flag(tmp_path):
    # A throwaway repo with one committed code file stands in for stacker-bot.
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    run_git(tmp_path, "config", "user.name", "test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    run_git(tmp_path, "add", ".")
    run_git(tmp_path, "commit", "-q", "-m", "start")

    # Clean tree: not flagged.
    _commit, flagged = evaluate_agents.git_state(tmp_path)
    assert flagged is False

    # A new, untracked result file: still not flagged (the M1 bug).
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run.json").write_text("{}\n")
    _commit, flagged = evaluate_agents.git_state(tmp_path)
    assert flagged is False

    # An edit to code: flagged.
    (tmp_path / "src" / "code.py").write_text("x = 2\n")
    _commit, flagged = evaluate_agents.git_state(tmp_path)
    assert flagged is True
