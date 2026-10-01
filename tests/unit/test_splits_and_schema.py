import numpy as np
import pandas as pd
import pytest

from drscreen.data.schema import (
    GRADE,
    GROUP,
    IMAGE_ID,
    IMAGE_PATH,
    SOURCE,
    SPLIT,
    validate_manifest,
)
from drscreen.data.splits import assign_splits
from drscreen.exceptions import DataError


def _manifest(n_groups: int = 200, images_per_group: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    group_labels = rng.choice(3, size=n_groups, p=[0.6, 0.3, 0.1])
    rows = [
        {
            IMAGE_ID: f"{g}_{i}",
            IMAGE_PATH: f"/img/{g}_{i}.png",
            GRADE: int(group_labels[g]),
            SOURCE: "test",
            GROUP: str(g),
        }
        for g in range(n_groups)
        for i in range(images_per_group)
    ]
    return pd.DataFrame(rows)


def test_groups_never_straddle_splits() -> None:
    result = assign_splits(_manifest(), GRADE, val_fraction=0.2, seed=1)

    assert (result.groupby(GROUP)[SPLIT].nunique() == 1).all()


def test_val_fraction_and_stratification_are_approximate() -> None:
    result = assign_splits(_manifest(), GRADE, val_fraction=0.2, seed=1)

    val = result[result[SPLIT] == "val"]
    assert len(val) / len(result) == pytest.approx(0.2, abs=0.03)
    overall = result[GRADE].value_counts(normalize=True)
    in_val = val[GRADE].value_counts(normalize=True)
    assert (overall - in_val).abs().max() < 0.05


def test_split_is_reproducible_and_seed_dependent() -> None:
    manifest = _manifest()

    first = assign_splits(manifest, GRADE, 0.2, seed=1)[SPLIT]
    again = assign_splits(manifest, GRADE, 0.2, seed=1)[SPLIT]
    other = assign_splits(manifest, GRADE, 0.2, seed=2)[SPLIT]

    assert first.equals(again)
    assert not first.equals(other)


def test_split_does_not_modify_input() -> None:
    manifest = _manifest()

    assign_splits(manifest, GRADE, 0.2, seed=1)

    assert SPLIT not in manifest.columns


@pytest.mark.parametrize("fraction", [0.0, 0.6, -0.1])
def test_invalid_val_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        assign_splits(_manifest(), GRADE, fraction, seed=1)


def test_valid_manifest_passes() -> None:
    validate_manifest(_manifest(), GRADE, num_classes=5)


def test_manifest_missing_column_is_rejected() -> None:
    with pytest.raises(DataError, match="missing required columns"):
        validate_manifest(_manifest().drop(columns=[GROUP]), GRADE, 5)


def test_manifest_null_is_rejected() -> None:
    manifest = _manifest()
    manifest.loc[0, GROUP] = None

    with pytest.raises(DataError, match="null"):
        validate_manifest(manifest, GRADE, 5)


def test_manifest_out_of_range_label_is_rejected() -> None:
    manifest = _manifest()
    manifest.loc[0, GRADE] = 5

    with pytest.raises(DataError, match=r"outside 0\.\.4"):
        validate_manifest(manifest, GRADE, 5)


def test_manifest_float_labels_are_rejected() -> None:
    manifest = _manifest()
    manifest[GRADE] = manifest[GRADE].astype(float)

    with pytest.raises(DataError, match="must be integer"):
        validate_manifest(manifest, GRADE, 5)


@pytest.mark.parametrize("column", [IMAGE_ID, IMAGE_PATH])
def test_manifest_duplicates_are_rejected(column: str) -> None:
    manifest = _manifest()
    manifest.loc[1, column] = manifest.loc[0, column]

    with pytest.raises(DataError, match="duplicated"):
        validate_manifest(manifest, GRADE, 5)
