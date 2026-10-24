"""Seeding of every random source and capture of run provenance.

A training run is reproducible when the same code, config and seed give the
same result. ``seed_everything`` covers Python, NumPy and torch (CPU and
CUDA); data-loader workers are seeded by torch from the loader's generator.
``git_state`` records which commit produced a run.
"""

import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

GIT_TIMEOUT_SECONDS = 10
COMMIT_FILE = "GIT_COMMIT"  # written beside a copy of the code that has no git history


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and torch, optionally forcing deterministic kernels.

    Args:
        seed: Seed applied to every generator.
        deterministic: If true, make cuDNN pick deterministic algorithms and
            disable autotuning. Runs become bit-for-bit repeatable on the
            same hardware, at some cost in speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    logger.info("Seeded all random sources with %d (deterministic=%s)", seed, deterministic)


def git_state(repo_dir: Path) -> dict[str, str | bool | None]:
    """Return the current commit and whether the working tree has local changes.

    Args:
        repo_dir: Any directory inside the repository.

    Returns:
        ``{"commit": sha or None, "dirty": bool or None}``. In a copy of the
        code without its history (such as a Kaggle upload) the commit comes
        from a ``GIT_COMMIT`` file in ``repo_dir`` and ``dirty`` is ``None``;
        both are ``None`` when neither git nor that file is available.
    """
    try:
        commit = _git(repo_dir, "rev-parse", "HEAD")
        dirty = bool(_git(repo_dir, "status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        commit_file = repo_dir / COMMIT_FILE
        if commit_file.is_file():
            return {"commit": commit_file.read_text(encoding="utf-8").strip(), "dirty": None}
        logger.warning("Git state unavailable; the run will not record its commit.")
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _git(repo_dir: Path, *args: str) -> str:
    """Run a git command and return its stripped standard output.

    Args:
        repo_dir: Working directory for the command.
        *args: Arguments to git.

    Returns:
        Standard output without surrounding whitespace.

    Raises:
        OSError: If git is not installed.
        subprocess.SubprocessError: If the command fails or times out.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    return result.stdout.strip()
