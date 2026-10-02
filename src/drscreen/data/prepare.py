"""Batch cropping and resizing of a manifest's images into an on-disk cache.

Decoding and cropping the multi-megapixel originals takes far longer than a
training step, so it runs once per dataset and training reads the small cached
images. Only the geometric steps are cached; lighting normalisation runs when
an image is loaded, after any augmentation (see ``drscreen.data.dataset``).

Each cache directory records the geometric recipe that produced it. Re-running
with the same recipe resumes where it stopped; running with a different recipe
fails instead of mixing images made with different settings.
"""

import json
import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import pandas as pd

from drscreen.data.image_io import read_rgb, write_rgb
from drscreen.data.preprocessing import PreprocessingConfig, crop_and_resize
from drscreen.data.schema import IMAGE_ID, IMAGE_PATH
from drscreen.exceptions import DataError

logger = logging.getLogger(__name__)

RECIPE_FILE = "preprocessing.json"
# Lighting normalisation is applied at load time, so it is not part of the cache.
UNCACHED_FIELDS = {"ben_graham"}
IMAGES_SUBDIR = "images"
PROCESSED_SUFFIX = ".png"  # lossless, so the cache adds no compression artefacts
ERROR_COLUMN = "error"
WORKER_CHUNK_SIZE = 16  # images per task sent to a worker; amortises process overhead


@dataclass(frozen=True)
class PrepareResult:
    """Outcome of preprocessing a manifest.

    Attributes:
        manifest: Input rows that succeeded, with ``image_path`` pointing at
            the preprocessed image.
        failures: Input rows that failed, with an ``error`` column.
    """

    manifest: pd.DataFrame
    failures: pd.DataFrame


def prepare_images(
    manifest: pd.DataFrame, output_dir: Path, config: PreprocessingConfig, workers: int
) -> PrepareResult:
    """Crop and resize every image in a manifest and write the results to disk.

    Images whose preprocessing raises ``DataError`` (unreadable files, no
    detectable fundus) are reported in ``PrepareResult.failures`` rather than
    aborting the batch; any other exception is a bug and propagates.

    Args:
        manifest: Validated manifest of raw images; outputs are named by ``image_id``.
        output_dir: Cache directory for this dataset and recipe.
        config: Preprocessing recipe.
        workers: Number of worker processes; 1 runs in-process.

    Returns:
        The successful and failed rows.

    Raises:
        DataError: If ``output_dir`` holds images made with a different recipe.
    """
    _claim_cache_dir(output_dir, config)
    sources = [Path(path) for path in manifest[IMAGE_PATH]]
    targets = [
        output_dir / IMAGES_SUBDIR / f"{image_id}{PROCESSED_SUFFIX}"
        for image_id in manifest[IMAGE_ID]
    ]

    errors = list(_run_all(sources, targets, config, workers))

    failed = [error is not None for error in errors]
    succeeded = manifest.loc[[not flag for flag in failed]].copy()
    succeeded[IMAGE_PATH] = [
        str(target) for target, flag in zip(targets, failed, strict=True) if not flag
    ]
    failures = manifest.loc[failed].copy()
    failures[ERROR_COLUMN] = [error for error in errors if error is not None]
    logger.info(
        "Preprocessed %d images into %s; %d failed.", len(succeeded), output_dir, len(failures)
    )
    return PrepareResult(succeeded.reset_index(drop=True), failures.reset_index(drop=True))


def _run_all(
    sources: list[Path], targets: list[Path], config: PreprocessingConfig, workers: int
) -> Iterator[str | None]:
    """Preprocess image pairs, in parallel when ``workers > 1``.

    Args:
        sources: Raw image paths.
        targets: Output paths, aligned with ``sources``.
        config: Preprocessing recipe.
        workers: Number of worker processes.

    Yields:
        For each pair, ``None`` on success or the error message on failure.
    """
    if workers == 1:
        yield from map(_process_one, sources, targets, repeat(config))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(
            _process_one, sources, targets, repeat(config), chunksize=WORKER_CHUNK_SIZE
        )


def _process_one(source: Path, target: Path, config: PreprocessingConfig) -> str | None:
    """Preprocess one image, skipping it if already cached.

    Args:
        source: Raw image path.
        target: Output path.
        config: Preprocessing recipe.

    Returns:
        ``None`` on success, otherwise the error message.
    """
    if target.is_file():
        return None
    try:
        write_rgb(crop_and_resize(read_rgb(source), config), target)
    except DataError as exc:
        return str(exc)
    return None


def cached_recipe(config: PreprocessingConfig) -> dict[str, object]:
    """Return the part of a recipe that determines cached images.

    Args:
        config: Full preprocessing recipe.

    Returns:
        JSON-compatible mapping of the geometric recipe fields.
    """
    return config.model_dump(mode="json", exclude=UNCACHED_FIELDS)


def read_cached_recipe(cache_dir: Path) -> dict[str, object]:
    """Read the recipe stored in a cache directory.

    Args:
        cache_dir: Directory written by ``prepare_images``.

    Returns:
        The stored geometric recipe.

    Raises:
        DataError: If the directory has no recipe file.
    """
    recipe_path = cache_dir / RECIPE_FILE
    if not recipe_path.is_file():
        raise DataError(f"No {RECIPE_FILE} in {cache_dir}; run `make prepare-data` first.")
    stored: dict[str, object] = json.loads(recipe_path.read_text(encoding="utf-8"))
    return stored


def _claim_cache_dir(output_dir: Path, config: PreprocessingConfig) -> None:
    """Record the recipe in a cache directory, or check it matches the stored one.

    Args:
        output_dir: Cache directory.
        config: Recipe about to be used.

    Raises:
        DataError: If the directory was produced with a different recipe.
    """
    recipe_path = output_dir / RECIPE_FILE
    recipe = cached_recipe(config)
    if recipe_path.is_file():
        stored = read_cached_recipe(output_dir)
        if stored != recipe:
            raise DataError(
                f"{output_dir} was produced with a different preprocessing recipe "
                f"({stored}). Delete the directory or point paths.processed_dir elsewhere."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
