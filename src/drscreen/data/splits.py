"""Reproducible, stratified, patient-grouped train/validation splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from drscreen.data.schema import GROUP, SPLIT, Split

MAX_VAL_FRACTION = 0.5


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
