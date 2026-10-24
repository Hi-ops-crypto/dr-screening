"""Reproducible, stratified, patient-grouped train/validation splits and subsampling."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from drscreen.data.schema import GROUP, SPLIT, Split

MAX_VAL_FRACTION = 0.5


def cap_per_class(
    manifest: pd.DataFrame, label_column: str, max_per_class: int, seed: int
) -> pd.DataFrame:
    """Randomly keep at most ``max_per_class`` rows of each class.

    Args:
        manifest: Manifest to subsample.
        label_column: Column holding the class.
        max_per_class: Largest number of rows kept per class.
        seed: Seed for the sample; the same seed keeps the same rows.

    Returns:
        The kept rows in their original order.
    """
    shuffled = manifest.sample(frac=1, random_state=seed)
    kept = shuffled.groupby(label_column).head(max_per_class)
    return manifest.loc[sorted(kept.index)].reset_index(drop=True)


def assign_splits(
    manifest: pd.DataFrame, label_column: str, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Add a ``split`` column assigning each row to train or validation.

    Rows sharing a ``group`` always land in the same split, and class
    proportions are kept approximately equal across splits. The validation
    set is the first fold of a ``round(1 / val_fraction)``-fold
    ``StratifiedGroupKFold``, so the achieved fraction is approximate.

    Args:
        manifest: Manifest with ``label_column`` and ``group`` columns.
        label_column: Column to stratify on.
        val_fraction: Target fraction of rows in validation, in ``(0, 0.5]``.
        seed: Seed for the shuffle; the same seed gives the same split.

    Returns:
        A copy of ``manifest`` with a ``split`` column.

    Raises:
        ValueError: If ``val_fraction`` is outside ``(0, 0.5]``.
    """
    if not 0 < val_fraction <= MAX_VAL_FRACTION:
        raise ValueError(f"val_fraction must be in (0, {MAX_VAL_FRACTION}], got {val_fraction}.")
    folds = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True, random_state=seed)
    _, val_index = next(folds.split(manifest, manifest[label_column], manifest[GROUP]))
    is_val = np.zeros(len(manifest), dtype=bool)
    is_val[val_index] = True
    result = manifest.copy()
    result[SPLIT] = np.where(is_val, Split.VAL.value, Split.TRAIN.value)
    return result
