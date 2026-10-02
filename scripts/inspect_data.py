"""Report class distributions and render per-class sample grids for each dataset.

Grids are rendered for both raw and preprocessed images (when available), so
cropping and normalisation can be checked by eye. Preprocessed grids show the
cached crops after lighting normalisation, as the model sees them.
"""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import pandas as pd

from drscreen.cli import build_parser, run
from drscreen.config import load_config
from drscreen.data.inspection import class_distribution, render_sample_grid
from drscreen.data.preprocessing import normalise_lighting
from drscreen.data.schema import CLASS_NAMES, LABEL_COLUMN, Source
from drscreen.data.settings import DataConfig
from drscreen.exceptions import DataError

DEFAULT_CONFIG = Path("configs/data.yaml")

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Write distribution CSVs and sample-grid PNGs to the reports directory.

    Args:
        args: Parsed command-line arguments.

    Raises:
        DataError: If no enabled dataset has a manifest yet.
    """
    config = load_config(args.config, DataConfig, args.override)
    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    inspected = 0
    for source in Source:
        if not config.sources.is_enabled(source):
            continue
        label_column = LABEL_COLUMN[source]
        names = CLASS_NAMES[label_column]
        lighting = partial(normalise_lighting, config=config.preprocessing.for_source(source))
        stages = {
            "raw": (config.paths.manifest(source), None),
            "processed": (config.paths.processed_manifest(source), lighting),
        }
        for stage, (manifest_path, transform) in stages.items():
            if not manifest_path.is_file():
                logger.info("No %s manifest for %s; skipping", stage, source.value)
                continue
            manifest = pd.read_csv(manifest_path)
            distribution = class_distribution(manifest, label_column, names)
            distribution.to_csv(reports_dir / f"{source.value}_{stage}_distribution.csv")
            logger.info("%s (%s):\n%s", source.value, stage, distribution.to_string())
            grid = render_sample_grid(
                manifest,
                label_column,
                names,
                samples_per_class=config.inspection.samples_per_class,
                thumbnail_size=config.inspection.thumbnail_size,
                seed=config.inspection.seed,
                transform=transform,
            )
            grid.savefig(reports_dir / f"{source.value}_{stage}_samples.png")
            inspected += 1
    if inspected == 0:
        raise DataError("No manifests found to inspect; run `make fetch-data` first.")
    logger.info("Reports written to %s", reports_dir)


if __name__ == "__main__":
    sys.exit(run(main, build_parser(__doc__, DEFAULT_CONFIG).parse_args()))
