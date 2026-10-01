"""Parsers that turn each dataset's on-disk layout into a manifest.

Each parser reads only label files and directory listings; it never decodes
images. Parsers fail loudly when labelled images are missing, except for EyeQ,
whose label files deliberately cover more images than the EyePACS mirror we
download (see ``parse_eyeq``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from drscreen.data.schema import (
    DR_GRADE_NAMES,
    GRADE,
    GROUP,
    IMAGE_ID,
    IMAGE_PATH,
    QUALITY,
    QUALITY_NAMES,
    SOURCE,
    Source,
    validate_manifest,
)
from drscreen.exceptions import DataError

logger = logging.getLogger(__name__)

APTOS_LABELS_FILE = "train.csv"
APTOS_IMAGES_DIR = "train_images"
APTOS_IMAGE_SUFFIX = ".png"
APTOS_ID_COLUMN = "id_code"
APTOS_GRADE_COLUMN = "diagnosis"

# The official release has a training and a testing subset whose image names
# overlap (both start at IDRiD_001), so each subset's labels must be paired
# with that subset's own image folder.
IDRID_SUBSETS = {"Training": "train", "Testing": "test"}
IDRID_LABELS_GLOB = "*IDRiD_Disease Grading_{subset} Labels.csv"
IDRID_IMAGES_DIR_GLOB = "*Disease Grading/*Original Images/*{subset} Set"
IDRID_IMAGE_SUFFIX = ".jpg"
IDRID_NAME_COLUMN = "Image name"
IDRID_GRADE_COLUMN = "Retinopathy grade"

EYEQ_IMAGE_COLUMN = "image"
EYEQ_QUALITY_COLUMN = "quality"
EYEPACS_EYE_SEPARATOR = "_"  # EyePACS files are named <patient id>_<left|right>.jpeg


def parse_aptos(root: Path) -> pd.DataFrame:
    """Build a manifest for the labelled APTOS 2019 training set.

    Args:
        root: Directory containing ``train.csv`` and ``train_images/``.

    Returns:
        A manifest with a ``grade`` label column. APTOS provides no patient
        identifiers, so each image is its own group.

    Raises:
        DataError: If the label file or any labelled image is missing.
    """
    labels = _read_csv(root / APTOS_LABELS_FILE)
    _require_columns(labels, [APTOS_ID_COLUMN, APTOS_GRADE_COLUMN], root / APTOS_LABELS_FILE)
    images_dir = root / APTOS_IMAGES_DIR
    image_ids = labels[APTOS_ID_COLUMN].astype(str)
    manifest = pd.DataFrame(
        {
            IMAGE_ID: image_ids,
            IMAGE_PATH: [
                str(images_dir / f"{image_id}{APTOS_IMAGE_SUFFIX}") for image_id in image_ids
            ],
            GRADE: labels[APTOS_GRADE_COLUMN].astype(int),
            SOURCE: Source.APTOS.value,
            GROUP: image_ids,
        }
    )
    _require_images_exist(manifest, Source.APTOS)
    validate_manifest(manifest, GRADE, len(DR_GRADE_NAMES))
    return manifest


def parse_idrid(root: Path) -> pd.DataFrame:
    """Build a manifest for all 516 IDRiD disease-grading images.

    Both the official training and testing subsets are included: IDRiD is used
    only as an external test set, so its internal split is irrelevant here.
    Files are located by pattern anywhere under ``root`` because mirrors nest
    the official folders differently.

    Args:
        root: Root of an IDRiD download containing the "B. Disease Grading" part.

    Returns:
        A manifest with a ``grade`` label column. Image ids are prefixed with
        the subset (``train_IDRiD_001``) because names repeat across subsets.

    Raises:
        DataError: If a subset's label file or image folder is not found
            exactly once, or a labelled image is missing.
    """
    subsets = [
        _parse_idrid_subset(root, subset, prefix) for subset, prefix in IDRID_SUBSETS.items()
    ]
    manifest = pd.concat(subsets, ignore_index=True)
    _require_images_exist(manifest, Source.IDRID)
    validate_manifest(manifest, GRADE, len(DR_GRADE_NAMES))
    return manifest


def _parse_idrid_subset(root: Path, subset: str, prefix: str) -> pd.DataFrame:
    """Parse one IDRiD subset's labels against that subset's image folder.

    Args:
        root: Root of the IDRiD download.
        subset: ``"Training"`` or ``"Testing"``, as named in the release.
        prefix: Short prefix for this subset's image ids.

    Returns:
        A manifest of the subset's images.

    Raises:
        DataError: If the label file or image folder is not found exactly once.
    """
    label_file = _find_one(root, IDRID_LABELS_GLOB.format(subset=subset))
    images_dir = _find_one(root, IDRID_IMAGES_DIR_GLOB.format(subset=subset))
    labels = _read_csv(label_file)
    labels.columns = labels.columns.str.strip()
    _require_columns(labels, [IDRID_NAME_COLUMN, IDRID_GRADE_COLUMN], label_file)
    labels = labels.dropna(subset=[IDRID_NAME_COLUMN])
    names = labels[IDRID_NAME_COLUMN].str.strip()
    image_ids = prefix + "_" + names
    return pd.DataFrame(
        {
            IMAGE_ID: image_ids,
            IMAGE_PATH: [str(images_dir / f"{name}{IDRID_IMAGE_SUFFIX}") for name in names],
            GRADE: labels[IDRID_GRADE_COLUMN].astype(int),
            SOURCE: Source.IDRID.value,
            GROUP: image_ids,
        }
    )


def parse_eyeq(label_files: Sequence[Path], images_dir: Path) -> pd.DataFrame:
    """Build a manifest of EyeQ quality labels joined to available EyePACS images.

    EyeQ labels cover images from both the EyePACS training and test sets,
    but the mirror we use contains only the training set. Labelled images
    that are absent are dropped and the count is logged.

    Args:
        label_files: EyeQ label CSVs with ``image`` and ``quality`` columns.
        images_dir: Directory containing EyePACS ``.jpeg`` images.

    Returns:
        A manifest with a ``quality`` label column, grouped by patient so that
        a patient's two eyes always land in the same split.

    Raises:
        DataError: If a label file is malformed or no labelled image is found.
    """
    labels = pd.concat([_read_csv(path) for path in label_files], ignore_index=True)
    _require_columns(labels, [EYEQ_IMAGE_COLUMN, EYEQ_QUALITY_COLUMN], label_files[0])
    manifest = pd.DataFrame(
        {
            IMAGE_ID: labels[EYEQ_IMAGE_COLUMN].map(lambda name: Path(name).stem),
            IMAGE_PATH: [str(images_dir / name) for name in labels[EYEQ_IMAGE_COLUMN]],
            QUALITY: labels[EYEQ_QUALITY_COLUMN].astype(int),
            SOURCE: Source.EYEQ.value,
            GROUP: labels[EYEQ_IMAGE_COLUMN].str.split(EYEPACS_EYE_SEPARATOR).str[0],
        }
    )
    present = manifest[IMAGE_PATH].map(lambda path: Path(path).is_file())
    if not present.any():
        raise DataError(f"None of the {len(manifest)} EyeQ-labelled images exist in {images_dir}.")
    logger.info(
        "EyeQ: %d of %d labelled images found in %s; %d absent from this mirror were dropped.",
        int(present.sum()),
        len(manifest),
        images_dir,
        int((~present).sum()),
    )
    manifest = manifest[present].reset_index(drop=True)
    validate_manifest(manifest, QUALITY, len(QUALITY_NAMES))
    return manifest


def _find_one(root: Path, pattern: str) -> Path:
    """Find the single path under ``root`` matching a recursive glob pattern.

    Args:
        root: Directory to search.
        pattern: Glob pattern, matched at any depth.

    Returns:
        The matching path.

    Raises:
        DataError: If there is not exactly one match.
    """
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise DataError(
            f"Expected exactly one path matching {pattern!r} under {root}, "
            f"found {len(matches)}: {matches}. Is the download complete?"
        )
    return matches[0]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, converting failures into ``DataError``.

    Args:
        path: CSV file path.

    Returns:
        The parsed table.

    Raises:
        DataError: If the file is missing or cannot be parsed.
    """
    if not path.is_file():
        raise DataError(f"Label file not found: {path}")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Label file {path} could not be parsed: {exc}") from exc


def _require_columns(table: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    """Check that a label table has the expected columns.

    Args:
        table: Parsed label table.
        columns: Required column names.
        path: Origin of the table, for the error message.

    Raises:
        DataError: If any column is missing.
    """
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise DataError(f"Label file {path} lacks columns {missing}; found {list(table.columns)}.")


def _require_images_exist(manifest: pd.DataFrame, source: Source) -> None:
    """Check that every image referenced by a manifest exists.

    Args:
        manifest: Manifest to check.
        source: Dataset name, for the error message.

    Raises:
        DataError: If any image file is missing.
    """
    missing = [path for path in manifest[IMAGE_PATH] if not Path(path).is_file()]
    if missing:
        raise DataError(
            f"{source.value}: {len(missing)} of {len(manifest)} labelled images are missing, "
            f"e.g. {missing[0]}. The download may be incomplete; re-run `make fetch-data`."
        )
