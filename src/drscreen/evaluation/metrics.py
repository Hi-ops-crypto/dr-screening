"""Classification metrics suited to ordinal grades and screening.

Accuracy is never the headline: with about half of images at grade 0, a model
that always answers 0 scores around 50% while missing every case that needs
referral. This module reports:

* Quadratic weighted kappa, which penalises a disagreement by the square of
  its distance in grades, so confusing grade 4 with 0 costs far more than
  confusing it with 3.
* Per-class sensitivity (share of true cases found) and specificity (share of
  non-cases correctly left alone), one class against the rest.
* Referable-DR sensitivity and specificity, treating every grade at or above
  a threshold as "refer". This is the decision a screening programme acts on.
* The same for a probability-threshold referral rule: refer when the
  predicted grade is referable or the summed probability of the referable
  grades reaches a threshold chosen on validation data. This trades
  specificity for sensitivity without retraining.
* Exact and within-one-grade accuracy, as secondary figures for readers who
  ask for "accuracy"; neither is used to select models.
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
Probabilities = npt.NDArray[np.float64]
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


def referral_decisions(probabilities: Probabilities, min_grade: int, threshold: float) -> Labels:
    """Apply the probability-threshold referral rule to every image.

    An image is referred when its most likely grade is at least
    ``min_grade`` or when the probabilities of grades ``min_grade`` and above
    sum to at least ``threshold``. The serving code in
    ``drscreen.inference.referral`` applies the same rule.

    Args:
        probabilities: Softmax rows, one per image.
        min_grade: Lowest grade counted as referable.
        threshold: Referable probability at which to refer.

    Returns:
        ``min_grade`` for referred images and 0 otherwise, so the result can
        be passed to ``referable_sensitivity_specificity`` as predictions.
    """
    refer = (probabilities.argmax(axis=1) >= min_grade) | (
        probabilities[:, min_grade:].sum(axis=1) >= threshold
    )
    decisions: Labels = np.where(refer, min_grade, 0).astype(np.int64)
    return decisions


def choose_referral_threshold(
    y_true: Labels, probabilities: Probabilities, min_grade: int, target_sensitivity: float
) -> float:
    """Return the highest threshold whose referral rule reaches a target sensitivity.

    Higher thresholds refer fewer images, so the highest one that still
    reaches the target keeps specificity as high as possible.

    Args:
        y_true: True grades of the tuning images.
        probabilities: Softmax rows for the same images.
        min_grade: Lowest grade counted as referable.
        target_sensitivity: Required share of referable images referred.

    Returns:
        The threshold; 1.0 if the predicted grade alone reaches the target.

    Raises:
        ValueError: If the tuning images contain no referable case.
    """
    positive = y_true >= min_grade
    if not positive.any():
        raise ValueError(f"No image of grade {min_grade} or above to tune the threshold on.")
    already_referred = probabilities[positive].argmax(axis=1) >= min_grade
    still_needed = math.ceil(target_sensitivity * int(positive.sum())) - int(already_referred.sum())
    if still_needed <= 0:
        return 1.0
    candidates = np.sort(probabilities[positive][~already_referred, min_grade:].sum(axis=1))[::-1]
    return float(candidates[still_needed - 1])


def threshold_referral_metrics(
    y_true: Labels,
    probabilities: Probabilities,
    min_grade: int,
    threshold: float,
    bootstrap: BootstrapConfig | None,
) -> dict[str, float | None]:
    """Sensitivity and specificity of the probability-threshold referral rule.

    Args:
        y_true: True grades.
        probabilities: Softmax rows, one per image.
        min_grade: Lowest grade counted as referable.
        threshold: Referable probability at which to refer.
        bootstrap: Settings for confidence intervals, or ``None`` to skip them.

    Returns:
        Metric name to value under ``referable@threshold/``; undefined values
        are ``None``.
    """
    decisions = referral_decisions(probabilities, min_grade, threshold)
    sensitivity, specificity = referable_sensitivity_specificity(y_true, decisions, min_grade)
    metrics = {
        "referable@threshold/threshold": threshold,
        "referable@threshold/sensitivity": sensitivity,
        "referable@threshold/specificity": specificity,
        "referable@threshold/referred_fraction": float((decisions >= min_grade).mean()),
    }
    if bootstrap is not None:
        named: dict[str, MetricFn] = {
            "sensitivity": lambda t, p: referable_sensitivity_specificity(t, p, min_grade)[0],
            "specificity": lambda t, p: referable_sensitivity_specificity(t, p, min_grade)[1],
        }
        for name, metric in named.items():
            lower, upper = bootstrap_interval(y_true, decisions, metric, bootstrap)
            metrics[f"referable@threshold/{name}/ci_lower"] = lower
            metrics[f"referable@threshold/{name}/ci_upper"] = upper
    return {name: None if math.isnan(value) else value for name, value in metrics.items()}


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
    metrics: dict[str, float] = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes),
        "accuracy/exact": float((y_true == y_pred).mean()),
        "accuracy/within_one": float((np.abs(y_true - y_pred) <= 1).mean()),
    }
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
