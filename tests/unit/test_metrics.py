import numpy as np
import pytest

from drscreen.evaluation.metrics import (
    BootstrapConfig,
    bootstrap_interval,
    confusion,
    quadratic_weighted_kappa,
    referable_sensitivity_specificity,
    sensitivity_specificity,
    summarise,
)

NAMES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")


def _labels(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def test_perfect_agreement_gives_kappa_one() -> None:
    y = _labels(0, 1, 2, 3, 4, 0, 2)

    assert quadratic_weighted_kappa(y, y, 5) == pytest.approx(1.0)


def test_distant_errors_cost_more_than_near_errors() -> None:
    y_true = _labels(0, 1, 2, 3, 4, 4)
    near = _labels(0, 1, 2, 3, 4, 3)
    far = _labels(0, 1, 2, 3, 4, 0)

    assert quadratic_weighted_kappa(y_true, near, 5) > quadratic_weighted_kappa(y_true, far, 5)


def test_kappa_undefined_for_single_class() -> None:
    y = _labels(0, 0, 0)

    assert np.isnan(quadratic_weighted_kappa(y, y, 5))


def test_sensitivity_specificity_from_known_matrix() -> None:
    # true class 0: 3 correct, 1 predicted as 1; true class 1: 2 correct
    matrix = confusion(_labels(0, 0, 0, 0, 1, 1), _labels(0, 0, 0, 1, 1, 1), 3)

    sensitivities, specificities = sensitivity_specificity(matrix)

    assert sensitivities[0] == pytest.approx(3 / 4)
    assert sensitivities[1] == pytest.approx(1.0)
    assert np.isnan(sensitivities[2])  # no true cases of class 2
    assert specificities[0] == pytest.approx(1.0)
    assert specificities[1] == pytest.approx(3 / 4)
    assert specificities[2] == pytest.approx(1.0)


def test_referable_metrics_use_threshold() -> None:
    y_true = _labels(0, 1, 2, 3, 4, 2)
    y_pred = _labels(0, 2, 1, 3, 4, 2)  # one false referral, one missed referral

    sensitivity, specificity = referable_sensitivity_specificity(y_true, y_pred, min_grade=2)

    assert sensitivity == pytest.approx(3 / 4)
    assert specificity == pytest.approx(1 / 2)


def test_bootstrap_interval_brackets_estimate_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 200)
    y_pred = np.clip(y_true + rng.integers(-1, 2, 200), 0, 4)
    config = BootstrapConfig(resamples=200, confidence=0.95, seed=1)

    def metric(t: np.ndarray, p: np.ndarray) -> float:
        return quadratic_weighted_kappa(t, p, 5)

    lower, upper = bootstrap_interval(y_true, y_pred, metric, config)

    assert lower <= metric(y_true, y_pred) <= upper
    assert (lower, upper) == bootstrap_interval(y_true, y_pred, metric, config)


def test_summarise_reports_none_for_undefined_metrics() -> None:
    y = _labels(0, 0, 1, 1)

    metrics = summarise(y, y, NAMES, referable_min_grade=2, bootstrap=None)

    assert metrics["qwk"] == pytest.approx(1.0)
    assert metrics["sensitivity/no_dr"] == pytest.approx(1.0)
    assert metrics["sensitivity/proliferative_dr"] is None
    assert metrics["referable/sensitivity"] is None  # no referable cases
    assert metrics["referable/specificity"] == pytest.approx(1.0)


def test_summarise_adds_intervals_when_bootstrapping() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 100)
    config = BootstrapConfig(resamples=50, confidence=0.9, seed=0)

    metrics = summarise(y_true, y_true, NAMES, referable_min_grade=2, bootstrap=config)

    assert {"qwk/ci_lower", "qwk/ci_upper", "referable/sensitivity/ci_lower"} <= set(metrics)


def test_summarise_without_referral_threshold_omits_referable_metrics() -> None:
    y = _labels(0, 1, 2)

    metrics = summarise(
        y, y, ("Good", "Usable", "Reject"), referable_min_grade=None, bootstrap=None
    )

    assert not any(name.startswith("referable") for name in metrics)
