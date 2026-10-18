import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_server


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.com")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("alpha\n")
    (repo / "gone.txt").write_text("gone\n")
    (repo / "bin.dat").write_bytes(bytes(range(256)))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def start(repo: Path, config: Path = ROOT / "config.yaml") -> None:
    tool_server.RUN = tool_server.open_run(
        [
            *("--repo", str(repo)),
            *("--run-dir", str(repo.parent / "run")),
            *("--config", str(config)),
        ]
    )


def call(tool, **kwargs) -> tuple[bool, dict]:
    result = anyio.run(functools.partial(tool, **kwargs))
    return result.is_error, json.loads(result.content[0].text)


def test_file_tools_reject_paths_outside_the_repository(repo):
    start(repo)
    outside = repo.parent / "outside"
    outside.mkdir()
    (repo / "link").symlink_to(outside)
    config = (repo / ".git" / "config").read_bytes()
    for filepath in [
        "../x",
        str(repo / "a.txt"),
        "link/x",
        ".git/config",
        ".GIT/config",
    ]:
        for tool, kwargs in [
            (tool_server.write_file, {"content": "x"}),
            (tool_server.edit_file, {"old_string": "alpha", "new_string": "x"}),
        ]:
            is_error, data = call(tool, filepath=filepath, **kwargs)
            assert is_error and data["type"] == "PermissionError", (filepath, data)
    assert list(outside.iterdir()) == []
    assert not (repo.parent / "x").exists()
    assert (repo / ".git" / "config").read_bytes() == config
    assert (repo / "a.txt").read_text() == "alpha\n"


def test_edit_file_rejects_an_ambiguous_match_unless_allow_multiple(repo):
    start(repo)
    target = repo / "a.txt"
    target.write_text("x = 1\nx = 1\n")
    is_error, _ = call(
        tool_server.edit_file, filepath="a.txt", old_string="x = 1", new_string="x = 2"
    )
    assert is_error
    assert target.read_text() == "x = 1\nx = 1\n"
    is_error, data = call(
        tool_server.edit_file,
        filepath="a.txt",
        old_string="x = 1",
        new_string="x = 2",
        allow_multiple=True,
    )
    assert not is_error and data["replacements"] == 2
    assert target.read_text() == "x = 2\nx = 2\n"


def test_submit_patch_reproduces_the_tree_without_touching_the_real_index(
    repo, tmp_path
):
    start(repo)
    (repo / "a.txt").write_text("alpha\nbeta\n")
    git(repo, "add", "a.txt")
    (repo / "new.txt").write_text("new\n")
    (repo / "gone.txt").unlink()
    (repo / "bin.dat").write_bytes(bytes(reversed(range(256))))
    (repo / "new.bin").write_bytes(b"\x00\xff" * 64)
    index = (repo / ".git" / "index").read_bytes()
    is_error, data = call(tool_server.submit_patch)
    assert not is_error
    assert (repo / ".git" / "index").read_bytes() == index
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "-q", str(repo), str(fresh))
    git(fresh, "apply", data["patch_path"])
    for name in ["a.txt", "new.txt", "bin.dat", "new.bin"]:
        assert (fresh / name).read_bytes() == (repo / name).read_bytes()
    assert not (fresh / "gone.txt").exists()


def test_command_timeout_kills_background_children(repo):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["tools"]["command_timeout_s"] = 1
    short_timeout = repo.parent / "config.yaml"
    short_timeout.write_text(yaml.safe_dump(config))
    start(repo, short_timeout)
    pidfile = repo.parent / "sleep.pid"
    is_error, data = call(
        tool_server.run_command, command=f"sleep 300 & echo $! > {pidfile}; wait"
    )
    assert not is_error and data["timed_out"] and data["exit_code"] is None
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"sleep {pid} outlived the command timeout")


def test_restart_keeps_the_baseline_on_a_dirty_tree(repo):
    start(repo)
    baseline = git(repo, "rev-parse", "HEAD").decode().strip()
    call(tool_server.get_status)
    (repo / "a.txt").write_text("changed\n")
    git(repo, "commit", "-q", "-am", "move head")
    (repo / "new.txt").write_text("dirty\n")
    start(repo)
    is_error, data = call(tool_server.get_status)
    assert not is_error
    assert data["baseline_commit"] == baseline
    assert data["head_commit"] != baseline
    assert data["tool_calls"] == 2
    assert data["changed_paths"] == ["a.txt", "new.txt"]
