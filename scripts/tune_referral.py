"""Choose the referral threshold of grading checkpoints (an ensemble if several) on validation."""

import argparse
import sys
from pathlib import Path

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.settings import DataConfig
from drscreen.evaluation.external import EvaluationConfig, tune_referral_threshold

DEFAULT_CONFIG = Path("configs/evaluate.yaml")


def main(args: argparse.Namespace) -> None:
    """Load configs, tune the threshold and print the settings to copy.

    Args:
        args: Parsed command-line arguments.
    """
    config = load_config(args.config, EvaluationConfig, args.override)
    data = load_config(config.data_config, DataConfig)
    output_dir = args.output_dir or args.checkpoint[0].parent
    result = tune_referral_threshold(args.checkpoint, config, data, output_dir)
    checkpoints = " ".join(f"--checkpoint {path}" for path in args.checkpoint)
    print(
        "Put this under screener in configs/app.yaml:\n"
        f"  grading: {{run_ids: {result['grading_run_ids']}, flip_tta: {config.flip_tta}, "
        f"referral_threshold: {result['threshold']:.4f}}}\n"
        "and report it externally with:\n"
        f"  python scripts/evaluate.py {checkpoints} --output-dir {output_dir} "
        f"--override flip_tta={config.flip_tta} "
        f"--override referral.threshold={result['threshold']:.4f}"
    )


if __name__ == "__main__":
    parser = build_parser(__doc__, DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Path to a best.pt file; repeat to tune an ensemble.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Where to write the result; default: next to the first."
    )
    sys.exit(run(main, parser.parse_args()))
