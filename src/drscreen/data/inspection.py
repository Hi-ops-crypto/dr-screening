"""Dataset inspection: class distributions and per-class sample grids.

These reports are for a human to check that labels, splits and preprocessing
look right before any model is trained.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from drscreen.data.image_io import read_rgb
from drscreen.data.preprocessing import Image
from drscreen.data.schema import IMAGE_PATH, SPLIT

TOTAL_LABEL = "total"
PERCENT_DECIMALS = 1
GRID_CELL_INCHES = 1.6  # width and height of one thumbnail in the figure
PERCENT = 100


def class_distribution(
    manifest: pd.DataFrame, label_column: str, class_names: Sequence[str]
) -> pd.DataFrame:
    """Count images per class and per split.

    Args:
        manifest: Manifest with ``label_column`` and ``split`` columns.
        label_column: Column holding integer class labels.
        class_names: Display name for each label value.

    Returns:
        One row per class (including classes with no images) with a count
        column per split, a ``total`` column and a ``percent`` column.
    """
    counts = pd.crosstab(manifest[label_column], manifest[SPLIT])
    counts = counts.reindex(range(len(class_names)), fill_value=0)
    counts[TOTAL_LABEL] = counts.sum(axis=1)
    counts["percent"] = (PERCENT * counts[TOTAL_LABEL] / counts[TOTAL_LABEL].sum()).round(
        PERCENT_DECIMALS
    )
    counts.index = pd.Index(list(class_names), name=label_column)
    counts.columns.name = None
    return counts


def render_sample_grid(
    manifest: pd.DataFrame,
    label_column: str,
    class_names: Sequence[str],
    *,
    samples_per_class: int,
    thumbnail_size: int,
    seed: int,
    transform: Callable[[Image], Image] | None = None,
) -> Figure:
    """Render a grid with one row of randomly sampled images per class.

    Args:
        manifest: Manifest with ``image_path`` and ``label_column`` columns.
        label_column: Column holding integer class labels.
        class_names: Display name for each label value, used as row titles.
        samples_per_class: Number of images per row.
        thumbnail_size: Side length each image is resized to, in pixels.
        seed: Seed for choosing which images are shown.
        transform: Optional function applied to each image before display,
            such as lighting normalisation of cached images.

    Returns:
        A matplotlib figure; classes with fewer images leave blank cells.

    Raises:
        DataError: If a sampled image cannot be read.
    """
    rng = np.random.default_rng(seed)
    figure = Figure(
        figsize=(GRID_CELL_INCHES * samples_per_class, GRID_CELL_INCHES * len(class_names))
    )
    axes = figure.subplots(len(class_names), samples_per_class, squeeze=False)
    for label, name in enumerate(class_names):
        paths = manifest.loc[manifest[label_column] == label, IMAGE_PATH].to_numpy()
        chosen = rng.choice(paths, size=min(samples_per_class, len(paths)), replace=False)
        for column, axis in enumerate(axes[label]):
            axis.set_axis_off()
            if column < len(chosen):
                image = read_rgb(Path(chosen[column]))
                if transform is not None:
                    image = transform(image)
                thumbnail = cv2.resize(
                    image, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA
                )
                axis.imshow(thumbnail)
        axes[label][0].set_title(f"{label}: {name} (n={len(paths)})", loc="left", fontsize="small")
    figure.tight_layout()
    return figure
