"""Train a classifier described by a training config (grading or quality)."""

import argparse
import logging
import sys
from pathlib import Path

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.settings import DataConfig
from drscreen.training.settings import TrainingRunConfig
from drscreen.training.trainer import run_training

DEFAULT_CONFIG = Path("configs/train_grading.yaml")
REPO_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Load configs and train.

    Args:
        args: Parsed command-line arguments.
    """
    config = load_config(args.config, TrainingRunConfig, args.override)
    data = load_config(config.data_config, DataConfig)
    run_dir = run_training(config, data, REPO_DIR)
    logger.info("Training finished; outputs in %s", run_dir)


if __name__ == "__main__":
    sys.exit(run(main, build_parser(__doc__, DEFAULT_CONFIG).parse_args()))
