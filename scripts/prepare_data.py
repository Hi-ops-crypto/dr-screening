"""Preprocess every fetched dataset into the cache that training reads."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.prepare import prepare_images
from drscreen.data.schema import Source
from drscreen.data.settings import DataConfig
from drscreen.exceptions import DataError

DEFAULT_CONFIG = Path("configs/data.yaml")
FAILURES_FILE = "failures.csv"

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Preprocess each enabled dataset and write its processed manifest.

    Args:
        args: Parsed command-line arguments.

    Raises:
        DataError: If an enabled dataset has not been fetched yet.
    """
    config = load_config(args.config, DataConfig, args.override)
    for source in Source:
        if not config.sources.is_enabled(source):
            continue
        manifest_path = config.paths.manifest(source)
        if not manifest_path.is_file():
            raise DataError(f"No manifest at {manifest_path}; run `make fetch-data` first.")
        output = config.paths.processed_manifest(source)
        result = prepare_images(
            pd.read_csv(manifest_path),
            output.parent,
            config.preprocessing.for_source(source),
            config.prepare.workers,
        )
        result.manifest.to_csv(output, index=False)
        result.failures.to_csv(output.parent / FAILURES_FILE, index=False)
        if not result.failures.empty:
            logger.warning(
                "%s: %d images failed preprocessing; see %s",
                source.value,
                len(result.failures),
                output.parent / FAILURES_FILE,
            )


if __name__ == "__main__":
    sys.exit(run(main, build_parser(__doc__, DEFAULT_CONFIG).parse_args()))
