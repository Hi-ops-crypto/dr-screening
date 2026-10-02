"""Write evaluation results to disk: JSON metrics and a confusion-matrix plot."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

METRICS_FILE = "metrics.json"
CONFUSION_FILE = "confusion_matrix.png"
FIGURE_INCHES = 6
DARK_CELL_FRACTION = 0.5  # cells above this share of the max get white text


def write_report(
    output_dir: Path,
    metrics: dict[str, float | None],
    matrix: npt.NDArray[np.int64],
    class_names: Sequence[str],
) -> None:
    """Save metrics as JSON and the confusion matrix as an annotated heatmap.

    Args:
        output_dir: Directory to write into; created if missing.
        metrics: Output of ``evaluation.metrics.summarise``.
        matrix: Confusion matrix with rows as true and columns as predicted class.
        class_names: Display name for each class.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metrics": metrics, "confusion_matrix": matrix.tolist(), "classes": list(class_names)}
    (output_dir / METRICS_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure = Figure(figsize=(FIGURE_INCHES, FIGURE_INCHES))
    axis = figure.subplots()
    axis.imshow(matrix, cmap="Blues")
    ticks = range(len(class_names))
    axis.set_xticks(ticks, class_names, rotation=45, ha="right")
    axis.set_yticks(ticks, class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    threshold = matrix.max() * DARK_CELL_FRACTION
    for row in ticks:
        for column in ticks:
            value = matrix[row, column]
            colour = "white" if value > threshold else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=colour)
    figure.tight_layout()
    figure.savefig(output_dir / CONFUSION_FILE)
