"""Points-based screening-priority tier.

Each factor contributes the points of the highest band its value reaches (see
``configs/risk.yaml`` for the bands and their sources), and the total selects a
high, medium or low tier. The result lists every factor that added points, so
the tier can always be traced back to its inputs.

When only fasting glucose is known, HbA1c is estimated from it with the ADAG
regression, estimated average glucose (mg/dL) = 28.7 x HbA1c - 46.7 (Nathan DM
et al., Diabetes Care 2008;31(8):1473-8). That regression relates HbA1c to
average glucose, not fasting glucose, so the estimate is approximate and the
result says so.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, model_validator

from drscreen.config import StrictModel

ADAG_SLOPE = 28.7
ADAG_INTERCEPT = 46.7


class Band(StrictModel):
    """Values at or above ``min`` add ``points``."""

    min: float
    points: int = Field(gt=0)


class TierCutoffs(StrictModel):
    """Minimum total points for the high and medium tiers."""

    high_min_points: int = Field(gt=0)
    medium_min_points: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "TierCutoffs":
        if self.medium_min_points >= self.high_min_points:
            raise ValueError("medium_min_points must be below high_min_points.")
        return self


class RiskConfig(StrictModel):
    """Root schema of ``configs/risk.yaml``."""

    diabetes_duration_years: list[Band]
    hba1c_percent: list[Band]
    systolic_bp_mmhg: list[Band]
    age_years: list[Band]
    kidney_disease_points: int = Field(ge=0)
    tiers: TierCutoffs

    @model_validator(mode="after")
    def _bands_ascending(self) -> "RiskConfig":
        for name in ("diabetes_duration_years", "hba1c_percent", "systolic_bp_mmhg", "age_years"):
            bands: list[Band] = getattr(self, name)
            if [band.min for band in bands] != sorted({band.min for band in bands}):
                raise ValueError(f"{name} bands must have strictly increasing min values.")
        return self


class ClinicalFactors(StrictModel):
    """One person's risk factors, as entered in the app.

    At least one of ``hba1c_percent`` and ``fasting_glucose_mg_dl`` is
    required; HbA1c is used when both are given. The bounds reject typos and
    unit mix-ups (for example glucose in mmol/L), not unusual patients.
    """

    diabetes_duration_years: float = Field(ge=0, le=100)
    hba1c_percent: float | None = Field(default=None, ge=3, le=20)
    fasting_glucose_mg_dl: float | None = Field(default=None, ge=40, le=600)
    systolic_bp_mmhg: float = Field(ge=60, le=260)
    age_years: float = Field(ge=0, le=120)
    kidney_disease: bool

    @model_validator(mode="after")
    def _consistent(self) -> "ClinicalFactors":
        if self.hba1c_percent is None and self.fasting_glucose_mg_dl is None:
            raise ValueError("Enter HbA1c or fasting glucose.")
        if self.diabetes_duration_years > self.age_years:
            raise ValueError("Diabetes duration cannot exceed age.")
        return self


class Tier(StrEnum):
    """Screening priority."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class RiskAssessment:
    """A tier and how it was reached.

    Attributes:
        tier: Screening priority.
        points: Total points.
        factors: ``(description, points)`` for each factor that added points.
        notes: Caveats about the inputs, such as an estimated HbA1c.
    """

    tier: Tier
    points: int
    factors: list[tuple[str, int]]
    notes: list[str]


def estimate_hba1c(fasting_glucose_mg_dl: float) -> float:
    """Approximate HbA1c from a glucose value with the inverse ADAG regression.

    Args:
        fasting_glucose_mg_dl: Glucose in mg/dL.

    Returns:
        HbA1c in percent.
    """
    return (fasting_glucose_mg_dl + ADAG_INTERCEPT) / ADAG_SLOPE


def assess(factors: ClinicalFactors, config: RiskConfig) -> RiskAssessment:
    """Score clinical factors into a screening-priority tier.

    Args:
        factors: Validated risk factors.
        config: Bands, points and tier cut-offs.

    Returns:
        The tier, total points and the factors that contributed.
    """
    notes = []
    if factors.hba1c_percent is not None:
        hba1c, hba1c_label = factors.hba1c_percent, "HbA1c"
    else:
        assert factors.fasting_glucose_mg_dl is not None  # guaranteed by ClinicalFactors
        hba1c, hba1c_label = estimate_hba1c(factors.fasting_glucose_mg_dl), "Estimated HbA1c"
        notes.append(
            f"HbA1c was estimated as {hba1c:.1f} % from fasting glucose "
            f"{factors.fasting_glucose_mg_dl:g} mg/dL; a measured HbA1c is more reliable."
        )
    scored = [
        _score(
            f"Diabetes for {factors.diabetes_duration_years:g} years",
            factors.diabetes_duration_years,
            config.diabetes_duration_years,
            "years",
        ),
        _score(f"{hba1c_label} {hba1c:.1f} %", hba1c, config.hba1c_percent, "%"),
        _score(
            f"Systolic blood pressure {factors.systolic_bp_mmhg:g} mmHg",
            factors.systolic_bp_mmhg,
            config.systolic_bp_mmhg,
            "mmHg",
        ),
        _score(f"Age {factors.age_years:g}", factors.age_years, config.age_years, "years"),
    ]
    if factors.kidney_disease and config.kidney_disease_points:
        scored.append((config.kidney_disease_points, "Kidney disease"))
    points = sum(score for score, _ in scored)
    contributing = [(line, score) for score, line in scored if score]
    if points >= config.tiers.high_min_points:
        tier = Tier.HIGH
    elif points >= config.tiers.medium_min_points:
        tier = Tier.MEDIUM
    else:
        tier = Tier.LOW
    return RiskAssessment(tier=tier, points=points, factors=contributing, notes=notes)


def _score(description: str, value: float, bands: Sequence[Band], unit: str) -> tuple[int, str]:
    """Return the points of the highest band ``value`` reaches, with a description.

    Args:
        description: What the value is, for the contributing-factor line.
        value: The measured value.
        bands: Bands in increasing order of ``min``.
        unit: Unit of ``value``, for the threshold in the description.

    Returns:
        ``(points, line)``; points are 0 when no band is reached.
    """
    reached = [band for band in bands if value >= band.min]
    if not reached:
        return 0, description
    return reached[-1].points, f"{description}, at least {reached[-1].min:g} {unit}"
