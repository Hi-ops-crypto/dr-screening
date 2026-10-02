"""Classification metrics suited to ordinal grades and screening.

Accuracy is deliberately absent: with about half of images at grade 0, a model
that always answers 0 scores around 50% while missing every case that needs
referral. Instead this module reports:

* Quadratic weighted kappa, which penalises a disagreement by the square of
  its distance in grades, so confusing grade 4 with 0 costs far more than
  confusing it with 3.
* Per-class sensitivity (share of true cases found) and specificity (share of
  non-cases correctly left alone), one class against the rest.
* Referable-DR sensitivity and specificity, treating every grade at or above
  a threshold as "refer". This is the decision a screening programme acts on.
* Optional bootstrap confidence intervals, which matter for small test sets.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from drscreen.config import StrictModel

Labels = npt.NDArray[np.int64]
MetricFn = Callable[[Labels, Labels], float]

PERCENTILE_SCALE = 100
MIN_CLASSES_FOR_KAPPA = 2  # kappa is undefined when only one class occurs


class BootstrapConfig(StrictModel):
    """Bootstrap resampling for confidence intervals."""

    resamples: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    seed: int


def confusion(y_true: Labels, y_pred: Labels, num_classes: int) -> npt.NDArray[np.int64]:
    """Return the confusion matrix with rows as true and columns as predicted class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes; absent classes still get a row and column.

    Returns:
        Integer matrix of shape ``(num_classes, num_classes)``.
    """
    matrix: npt.NDArray[np.int64] = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    ).astype(np.int64)
    return matrix


def quadratic_weighted_kappa(y_true: Labels, y_pred: Labels, num_classes: int) -> float:
    """Return Cohen's kappa with quadratic weights.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of ordinal classes.

    Returns:
        Kappa in ``[-1, 1]``, or NaN when it is undefined (for example when
        both raters use a single identical class).
    """
    if len(np.union1d(y_true, y_pred)) < MIN_CLASSES_FOR_KAPPA:
        return math.nan
    return float(
        cohen_kappa_score(y_true, y_pred, labels=list(range(num_classes)), weights="quadratic")
    )


def sensitivity_specificity(matrix: npt.NDArray[np.int64]) -> tuple[list[float], list[float]]:
    """Compute one-vs-rest sensitivity and specificity for every class.

    Args:
        matrix: Confusion matrix from ``confusion``.

    Returns:
        ``(sensitivities, specificities)``, one value per class; NaN where a
        class has no true cases (sensitivity) or no non-cases (specificity).
    """
    total = matrix.sum()
    true_positive = np.diag(matrix)
    positives = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    negatives = total - positives
    true_negative = total - positives - predicted + true_positive
    return _ratios(true_positive, positives), _ratios(true_negative, negatives)


def referable_sensitivity_specificity(
    y_true: Labels, y_pred: Labels, min_grade: int
) -> tuple[float, float]:
    """Sensitivity and specificity of the binary decision ``grade >= min_grade``.

    Args:
        y_true: True grades.
        y_pred: Predicted grades.
        min_grade: Lowest grade counted as referable.

    Returns:
        ``(sensitivity, specificity)``; NaN when undefined.
    """
    binary = confusion(
        (y_true >= min_grade).astype(np.int64), (y_pred >= min_grade).astype(np.int64), 2
    )
    sensitivities, specificities = sensitivity_specificity(binary)
    return sensitivities[1], specificities[1]


def bootstrap_interval(
    y_true: Labels, y_pred: Labels, metric: MetricFn, config: BootstrapConfig
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a paired metric.

    Images are resampled with replacement; resamples where the metric is
    undefined are skipped.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        metric: Function of ``(y_true, y_pred)``.
        config: Number of resamples, confidence level and seed.

    Returns:
        ``(lower, upper)`` bounds, or NaNs if no resample was defined.
    """
    rng = np.random.default_rng(config.seed)
    size = len(y_true)
    values = []
    for _ in range(config.resamples):
        index = rng.integers(0, size, size)
        value = metric(y_true[index], y_pred[index])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    tail = (1 - config.confidence) / 2 * PERCENTILE_SCALE
    lower, upper = np.percentile(values, [tail, PERCENTILE_SCALE - tail])
    return float(lower), float(upper)


def summarise(
    y_true: Labels,
    y_pred: Labels,
    class_names: Sequence[str],
    referable_min_grade: int | None,
    bootstrap: BootstrapConfig | None,
) -> dict[str, float | None]:
    """Compute every reported metric as a flat, JSON-ready mapping.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Display name for each label value.
        referable_min_grade: Threshold for referable-DR metrics, or ``None``
            for tasks without a referral decision (the quality model).
        bootstrap: Settings for confidence intervals, or ``None`` to skip them.

    Returns:
        Metric name to value; undefined values are ``None``.
    """
    num_classes = len(class_names)
    metrics: dict[str, float] = {"qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes)}
    sensitivities, specificities = sensitivity_specificity(confusion(y_true, y_pred, num_classes))
    for name, sensitivity, specificity in zip(
        class_names, sensitivities, specificities, strict=True
    ):
        key = name.lower().replace(" ", "_")
        metrics[f"sensitivity/{key}"] = sensitivity
        metrics[f"specificity/{key}"] = specificity
    named: dict[str, MetricFn] = {
        "qwk": lambda t, p: quadratic_weighted_kappa(t, p, num_classes),
    }
    if referable_min_grade is not None:
        sensitivity, specificity = referable_sensitivity_specificity(
            y_true, y_pred, referable_min_grade
        )
        metrics["referable/sensitivity"] = sensitivity
        metrics["referable/specificity"] = specificity
        named["referable/sensitivity"] = lambda t, p: referable_sensitivity_specificity(
            t, p, referable_min_grade
        )[0]
        named["referable/specificity"] = lambda t, p: referable_sensitivity_specificity(
            t, p, referable_min_grade
        )[1]
    if bootstrap is not None:
        for name, metric in named.items():
            lower, upper = bootstrap_interval(y_true, y_pred, metric, bootstrap)
            metrics[f"{name}/ci_lower"] = lower
            metrics[f"{name}/ci_upper"] = upper
    return {name: None if math.isnan(value) else value for name, value in metrics.items()}


def _ratios(numerator: npt.NDArray[np.int64], denominator: npt.NDArray[np.int64]) -> list[float]:
    """Divide elementwise, giving NaN where the denominator is zero.

    Args:
        numerator: Counts.
        denominator: Counts.

    Returns:
        List of ratios.
    """
    return [
        float(top) / float(bottom) if bottom else math.nan
        for top, bottom in zip(numerator, denominator, strict=True)
    ]
