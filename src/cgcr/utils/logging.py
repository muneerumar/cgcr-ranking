"""Capture run provenance and configure Rich/file logging; implemented in ."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import re
import subprocess
import sys

from rich.console import Console
from rich.logging import RichHandler

from cgcr.utils.io import write_yaml


_active_run_dir: Path | None = None


def _git_snapshot() -> str:
    """Inspect the invocation directory, without changing repository state."""
    try:
        repository = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return "no-git\n"
    if repository.returncode != 0:
        return "no-git\n"
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else "no-commit"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    dirty = "true" if status.stdout.strip() else "false"
    return f"commit: {commit}\ndirty: {dirty}\n"


def _environment_snapshot() -> str:
    """Freeze packages with the same interpreter that is running CGCR."""
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    if freeze.returncode != 0:
        raise RuntimeError(
            "Cannot record pip freeze; install pip in the active environment. "
            + freeze.stderr.strip()
        )
    return (
        f"python: {sys.version}\n"
        f"executable: {sys.executable}\n"
        f"platform: {platform.platform()}\n"
        f"pip freeze:\n{freeze.stdout}"
    )


def start_run(
    exp_name: str, config: dict, results_root: str | Path = "results",
) -> Path:
    """Create a UTC-named run directory and its config, Git, env, and log files.

    The caller supplies an already resolved configuration including seeds.
    Git status is sampled before creating any run artifacts. Same-second
    name collisions receive a numeric suffix instead of overwriting a run.
    """
    if not isinstance(exp_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", exp_name,
    ):
        raise ValueError("exp_name must be a nonempty filename-safe name")
    if not isinstance(config, dict):
        raise TypeError("config must be a resolved dictionary")
    git_metadata = _git_snapshot()
    environment = _environment_snapshot()
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{exp_name}"
    collision = 0
    while True:
        suffix = f"_{collision:02d}" if collision else ""
        run_dir = root / f"{base_name}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            collision += 1
        else:
            break
    write_yaml(run_dir / "config.yaml", config)
    (run_dir / "git.txt").write_text(git_metadata, encoding="utf-8")
    (run_dir / "env.txt").write_text(environment, encoding="utf-8")
    global _active_run_dir
    _active_run_dir = run_dir.resolve()
    get_logger(run_dir=run_dir)
    return run_dir


def get_logger(
    run_dir: str | Path | None = None, name: str = "cgcr",
) -> logging.Logger:
    """Log to Rich on stderr and run.log, following the latest run by default.

    Call start_run first or pass an existing run directory explicitly.
    Repeated calls do not duplicate handlers; switching runs closes the
    previous CGCR-owned handlers on the same named logger.
    """
    selected = Path(run_dir) if run_dir is not None else _active_run_dir
    if selected is None:
        raise RuntimeError("Call start_run first or provide run_dir")
    log_path = (selected / "run.log").resolve()
    if not log_path.parent.is_dir():
        raise FileNotFoundError(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    managed = [
        handler for handler in logger.handlers
        if getattr(handler, "_cgcr_owned", False)
    ]
    if len(managed) == 2 and any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in managed
    ):
        return logger
    for handler in managed:
        logger.removeHandler(handler)
        handler.close()
    console = RichHandler(console=Console(stderr=True), markup=False)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    ))
    for handler in (console, file_handler):
        setattr(handler, "_cgcr_owned", True)
        logger.addHandler(handler)
    return logger
