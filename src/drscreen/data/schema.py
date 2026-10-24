"""The manifest schema every dataset is normalised to.

A manifest is a table with one row per image. Whatever a dataset's on-disk
layout, its parser in ``drscreen.data.sources`` produces a manifest with these
columns, so everything downstream handles every dataset identically.

Columns:
    image_id: Identifier unique within the dataset; names cached files.
    image_path: Path to the image file.
    <label column>: ``grade`` (ICDR 0-4) for DR datasets or ``quality``
        (EyeQ 0-2) for the quality dataset.
    source: Dataset name, one of ``Source``.
    group: Identifier of the patient or eye the image came from. Images that
        share a group are always placed in the same split, so one patient's
        two eyes cannot leak between training and validation.
    split: One of ``Split``; added by ``drscreen.data.splits``.
"""

from enum import StrEnum

import pandas as pd

from drscreen.exceptions import DataError

IMAGE_ID = "image_id"
IMAGE_PATH = "image_path"
GRADE = "grade"
QUALITY = "quality"
SOURCE = "source"
GROUP = "group"
SPLIT = "split"

# International Clinical Diabetic Retinopathy (ICDR) severity scale,
# Wilkinson et al., Ophthalmology 2003;110(9):1677-82. Index equals grade.
DR_GRADE_NAMES = (
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
)

# EyeQ quality labels, Fu et al., MICCAI 2019. Index equals label value.
QUALITY_NAMES = ("Good", "Usable", "Reject")


class Source(StrEnum):
    """Datasets the pipeline knows how to parse."""

    APTOS = "aptos"
    IDRID = "idrid"
    EYEQ = "eyeq"
    EYEPACS = "eyepacs"
    EYEPACS_TEST = "eyepacs_test"
    DDR = "ddr"


LABEL_COLUMN: dict[Source, str] = {
    Source.APTOS: GRADE,
    Source.IDRID: GRADE,
    Source.EYEQ: QUALITY,
    Source.EYEPACS: GRADE,
    Source.EYEPACS_TEST: GRADE,
    Source.DDR: GRADE,
}
CLASS_NAMES: dict[str, tuple[str, ...]] = {GRADE: DR_GRADE_NAMES, QUALITY: QUALITY_NAMES}


class Split(StrEnum):
    """Role of an image in model development."""

    TRAIN = "train"
    VAL = "val"
    EXTERNAL = "external"


def validate_manifest(manifest: pd.DataFrame, label_column: str, num_classes: int) -> None:
    """Check that a manifest has the required columns and valid labels.

    Args:
        manifest: The manifest to check.
        label_column: Name of the label column, ``GRADE`` or ``QUALITY``.
        num_classes: Labels must be integers in ``[0, num_classes)``.

    Raises:
        DataError: If a required column is missing, any value is null, labels
            are out of range, or image ids or paths are duplicated.
    """
    required = [IMAGE_ID, IMAGE_PATH, label_column, SOURCE, GROUP]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise DataError(f"Manifest is missing required columns: {missing}.")
    nulls = manifest[required].isna().any()
    if nulls.any():
        raise DataError(f"Manifest has null values in columns: {list(nulls[nulls].index)}.")
    labels = manifest[label_column]
    if not pd.api.types.is_integer_dtype(labels):
        raise DataError(f"Column {label_column!r} must be integer, got {labels.dtype}.")
    out_of_range = manifest[(labels < 0) | (labels >= num_classes)]
    if not out_of_range.empty:
        raise DataError(
            f"{len(out_of_range)} labels in {label_column!r} fall outside 0..{num_classes - 1}, "
            f"e.g. {out_of_range[IMAGE_PATH].iloc[0]} = {out_of_range[label_column].iloc[0]}."
        )
    for column in (IMAGE_ID, IMAGE_PATH):
        duplicated = manifest[column].duplicated()
        if duplicated.any():
            raise DataError(
                f"{int(duplicated.sum())} duplicated values in {column!r}, "
                f"e.g. {manifest.loc[duplicated, column].iloc[0]}."
            )
