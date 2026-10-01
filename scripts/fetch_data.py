"""Download APTOS 2019, IDRiD and EyeQ and write one manifest per dataset."""

import argparse
import logging
import sys
from pathlib import Path

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.fetch import fetch_manifest
from drscreen.data.schema import Source
from drscreen.data.settings import DataConfig

DEFAULT_CONFIG = Path("configs/data.yaml")

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Fetch every enabled dataset and save its manifest.

    Args:
        args: Parsed command-line arguments.
    """
    config = load_config(args.config, DataConfig, args.override)
    for source in Source:
        if not config.sources.is_enabled(source):
            logger.info("Skipping disabled source %s", source.value)
            continue
        manifest = fetch_manifest(source, config)
        path = config.paths.manifest(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        logger.info("Wrote %d rows to %s", len(manifest), path)


if __name__ == "__main__":
    sys.exit(run(main, build_parser(__doc__, DEFAULT_CONFIG).parse_args()))
