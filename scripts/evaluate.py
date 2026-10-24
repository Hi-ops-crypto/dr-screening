"""Evaluate trained checkpoints, as an ensemble, on the external test set (IDRiD by default)."""

import argparse
import sys
from pathlib import Path

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.settings import DataConfig
from drscreen.evaluation.external import EvaluationConfig, evaluate_checkpoint

DEFAULT_CONFIG = Path("configs/evaluate.yaml")


def main(args: argparse.Namespace) -> None:
    """Load configs and evaluate the given checkpoints.

    Args:
        args: Parsed command-line arguments.
    """
    config = load_config(args.config, EvaluationConfig, args.override)
    data = load_config(config.data_config, DataConfig)
    evaluate_checkpoint(args.checkpoint, config, data, args.output_dir or args.checkpoint[0].parent)


if __name__ == "__main__":
    parser = build_parser(__doc__, DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Path to a best.pt file; repeat to evaluate an ensemble.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Where to write the report; default: next to the first."
    )
    sys.exit(run(main, parser.parse_args()))
