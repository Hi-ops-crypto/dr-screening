from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from drscreen.config import load_config
from drscreen.exceptions import ConfigError
from drscreen.risk.calculator import ClinicalFactors, RiskConfig, Tier, assess, estimate_hba1c

RISK_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "risk.yaml"
CONFIG = load_config(RISK_CONFIG, RiskConfig)


def _factors(**changes: Any) -> ClinicalFactors:
    values: dict[str, Any] = {
        "diabetes_duration_years": 3,
        "hba1c_percent": 6.5,
        "systolic_bp_mmhg": 120,
        "age_years": 45,
        "kidney_disease": False,
    }
    return ClinicalFactors(**(values | changes))


def test_no_risk_factors_is_low_priority() -> None:
    result = assess(_factors(), CONFIG)

    assert result.tier is Tier.LOW
    assert result.points == 0
    assert result.factors == []
    assert result.notes == []


def test_highest_band_reached_counts_once() -> None:
    result = assess(_factors(diabetes_duration_years=25), CONFIG)

    assert result.points == 2
    assert result.tier is Tier.MEDIUM
    assert result.factors == [("Diabetes for 25 years, at least 20 years", 2)]


def test_band_minimum_is_inclusive() -> None:
    assert assess(_factors(systolic_bp_mmhg=139), CONFIG).points == 0
    assert assess(_factors(systolic_bp_mmhg=140), CONFIG).points == 1


def test_every_factor_at_its_highest_band_is_high_priority() -> None:
    result = assess(
        _factors(
            diabetes_duration_years=30,
            hba1c_percent=10,
            systolic_bp_mmhg=160,
            age_years=70,
            kidney_disease=True,
        ),
        CONFIG,
    )

    assert result.points == 7
    assert result.tier is Tier.HIGH
    assert len(result.factors) == 5
    assert result.factors[-1] == ("Kidney disease", 1)


def test_fasting_glucose_stands_in_for_missing_hba1c() -> None:
    result = assess(_factors(hba1c_percent=None, fasting_glucose_mg_dl=250), CONFIG)

    assert estimate_hba1c(250) == pytest.approx(10.34, abs=0.01)
    assert result.points == 2
    assert result.factors == [("Estimated HbA1c 10.3 %, at least 9.1 %", 2)]
    assert "estimated" in result.notes[0]


def test_measured_hba1c_wins_over_fasting_glucose() -> None:
    result = assess(_factors(hba1c_percent=6.0, fasting_glucose_mg_dl=400), CONFIG)

    assert result.points == 0
    assert result.notes == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hba1c_percent": None}, "HbA1c or fasting glucose"),
        ({"diabetes_duration_years": 50}, "cannot exceed age"),
        ({"fasting_glucose_mg_dl": 7.0, "hba1c_percent": None}, "greater than or equal to 40"),
        ({"systolic_bp_mmhg": -5}, "greater than or equal to 60"),
    ],
)
def test_implausible_inputs_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _factors(**changes)


def test_tier_cutoffs_must_be_ordered() -> None:
    with pytest.raises(ConfigError, match="medium_min_points"):
        load_config(RISK_CONFIG, RiskConfig, ["tiers.medium_min_points=4"])


def test_bands_must_increase() -> None:
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(
            RISK_CONFIG, RiskConfig, ["hba1c_percent=[{min: 9, points: 2}, {min: 7, points: 1}]"]
        )
