"""validate real run provenance, Git states, and console/file logging."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import re
import subprocess
import sys

import pytest
from rich.logging import RichHandler

from cgcr.utils.io import read_yaml
from cgcr.utils import logging as run_logging
from cgcr.utils.logging import get_logger, start_run


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_start_run_without_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"seed": 42, "ranker": {"temperature": 0}}
    run = start_run("smoke", config)
    assert run.parent == Path("results")
    assert re.fullmatch(r"\d{8}-\d{6}_smoke", run.name)
    assert {item.name for item in run.iterdir()} == {
        "config.yaml", "git.txt", "env.txt", "run.log",
    }
    assert read_yaml(run / "config.yaml") == config
    assert (run / "git.txt").read_text() == "no-git\n"
    environment = (run / "env.txt").read_text()
    assert sys.version in environment
    assert sys.executable in environment
    assert platform.platform() in environment
    assert "pip freeze:\n" in environment
    assert "numpy==" in environment
    assert "torch==" in environment


@pytest.mark.parametrize("dirty", [False, True])
def test_start_run_with_real_git_commit(tmp_path, monkeypatch, dirty):
    monkeypatch.chdir(tmp_path)
    git("init")
    (tmp_path / ".gitignore").write_text("results/\n")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n")
    git("add", ".gitignore", "tracked.txt")
    git(
        "-c", "user.name=T0 Test", "-c", "user.email=t0@example.invalid",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
        "commit", "-m", "test fixture",
    )
    commit = git("rev-parse", "HEAD")
    if dirty:
        tracked.write_text("modified\n")
    run = start_run("git-test", {"seed": 0})
    metadata = (run / "git.txt").read_text()
    assert f"commit: {commit}\n" in metadata
    assert f"dirty: {str(dirty).lower()}\n" in metadata


def test_start_run_with_unborn_git_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init")
    run = start_run("unborn", {"seed": 0})
    assert (run / "git.txt").read_text() == "commit: no-commit\ndirty: false\n"


def test_logger_writes_console_and_file_without_duplicate_handlers(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run = start_run("logging", {"seed": 0})
    logger = get_logger()
    assert get_logger() is logger
    assert sum(isinstance(h, RichHandler) for h in logger.handlers) == 1
    assert sum(isinstance(h, logging.FileHandler) for h in logger.handlers) == 1
    logger.info("T0 console and file message")
    assert "T0 console and file message" in capsys.readouterr().err
    assert (run / "run.log").read_text().count("T0 console and file message") == 1
    next_run = start_run("next-log", {"seed": 1})
    get_logger().info("second run only")
    assert "second run only" not in (run / "run.log").read_text()
    assert "second run only" in (next_run / "run.log").read_text()


def test_logger_requires_a_run(monkeypatch):
    monkeypatch.setattr(run_logging, "_active_run_dir", None)
    with pytest.raises(RuntimeError, match="start_run"):
        get_logger()


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "/absolute"])
def test_reject_unsafe_experiment_names(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="filename-safe"):
        start_run(name, {"seed": 0})
    assert not (tmp_path / "results").exists()


def test_same_second_runs_do_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FixedClock:
        @classmethod
        def now(cls, tz):
            return datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(run_logging, "datetime", FixedClock)
    first = start_run("repeat", {"seed": 1})
    second = start_run("repeat", {"seed": 2})
    assert first.name == "20261023-120000_repeat"
    assert second.name == "20261023-120000_repeat_01"
    assert read_yaml(first / "config.yaml") == {"seed": 1}
    assert read_yaml(second / "config.yaml") == {"seed": 2}


def test_freeze_failure_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    actual_run = subprocess.run

    def failed_freeze(command, **kwargs):
        if command == [sys.executable, "-m", "pip", "freeze"]:
            return subprocess.CompletedProcess(command, 1, "", "pip unavailable")
        return actual_run(command, **kwargs)

    monkeypatch.setattr(run_logging.subprocess, "run", failed_freeze)
    with pytest.raises(RuntimeError, match="pip unavailable"):
        start_run("broken-freeze", {})
    assert not (tmp_path / "results").exists()
