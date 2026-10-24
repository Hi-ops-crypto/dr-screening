"""Gradio interface: photograph screening and the risk calculator.

Handlers are plain functions of their inputs so they can be tested without a
browser; ``build_app`` only wires them to components.
"""

import logging
from pathlib import Path

import gradio as gr
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from pydantic import Field, ValidationError

from drscreen.app.audit import AuditLog
from drscreen.app.views import (
    CSS,
    EMPTY_RISK,
    EMPTY_SCREEN,
    HEAD,
    MASTHEAD,
    PHOTO_TIPS,
    THEME,
    render_risk,
    render_screening,
)
from drscreen.config import StrictModel, load_config
from drscreen.exceptions import InputError, ModelError
from drscreen.inference.screener import Screener, ScreenerConfig
from drscreen.risk.calculator import ClinicalFactors, RiskConfig, assess

logger = logging.getLogger(__name__)

TITLE = "DR screening"
FIELD_LABELS = {
    "diabetes_duration_years": "Years since diabetes diagnosis",
    "hba1c_percent": "HbA1c (%)",
    "fasting_glucose_mg_dl": "Fasting glucose (mg/dL)",
    "systolic_bp_mmhg": "Systolic blood pressure (mmHg)",
    "age_years": "Age (years)",
    "kidney_disease": "Diagnosed kidney disease",
}
# Gradio keeps uploads in a temporary cache; delete them every minute once a
# minute old, so photographs do not accumulate on disk.
UPLOAD_CACHE_SECONDS = 60
COMMIT_SHA = r"^[0-9a-f]{40}$"


class HubSource(StrictModel):
    """A Hugging Face Hub model repository pinned to one commit."""

    repo_id: str
    revision: str = Field(pattern=COMMIT_SHA)


class CheckpointSources(StrictModel):
    """Where the checkpoints come from.

    Without ``hub``, ``quality`` and ``grading`` are local paths; with it,
    they are file names inside the pinned Hub repository. Several grading
    checkpoints are served as an ensemble.
    """

    hub: HubSource | None
    quality: str
    grading: list[str] = Field(min_length=1)

    def resolve(self) -> tuple[Path, list[Path]]:
        """Return local paths of every checkpoint, downloading them if needed.

        Returns:
            ``(quality_path, grading_paths)``.

        Raises:
            ModelError: If a Hub download fails.
        """
        if self.hub is None:
            return Path(self.quality), [Path(name) for name in self.grading]
        hub = self.hub
        try:
            quality, *grading = (
                Path(hf_hub_download(hub.repo_id, name, revision=hub.revision))
                for name in (self.quality, *self.grading)
            )
        except (OSError, EntryNotFoundError) as exc:
            raise ModelError(
                f"Could not download checkpoints from {hub.repo_id}@{hub.revision}: {exc}"
            ) from exc
        return quality, grading


class AppConfig(StrictModel):
    """Root schema of ``configs/app.yaml``."""

    checkpoints: CheckpointSources
    screener: ScreenerConfig
    risk_config: Path
    audit_log: Path


def screen_photo(screener: Screener, audit: AuditLog, image: npt.NDArray[np.generic] | None) -> str:
    """Screen an upload, record the outcome and render it.

    Args:
        screener: Loaded models.
        audit: Where to record the outcome.
        image: Decoded upload.

    Returns:
        HTML for the result panel: the decision, grade scale and images, or
        a retake request.

    Raises:
        gr.Error: If the upload cannot be screened at all.
    """
    try:
        result = screener.screen(image)
    except InputError as exc:
        raise gr.Error(str(exc)) from exc
    audit.record(result.quality.label, None if result.grade is None else result.grade.grade)
    return render_screening(result)


def assess_risk(  # noqa: PLR0917  Gradio passes form values positionally
    config: RiskConfig,
    diabetes_duration_years: float | None,
    hba1c_percent: float | None,
    fasting_glucose_mg_dl: float | None,
    systolic_bp_mmhg: float | None,
    age_years: float | None,
    kidney_disease: bool,
) -> str:
    """Score form values into a priority tier and render it.

    Args:
        config: Risk bands and cut-offs.
        diabetes_duration_years: Years since diagnosis.
        hba1c_percent: HbA1c, or ``None`` if unknown.
        fasting_glucose_mg_dl: Fasting glucose, or ``None`` if unknown.
        systolic_bp_mmhg: Systolic blood pressure.
        age_years: Age.
        kidney_disease: Whether the person has kidney disease.

    Returns:
        HTML for the result panel: tier, points and contributing factors.

    Raises:
        gr.Error: If a value is missing or implausible, naming the form field.
    """
    try:
        factors = ClinicalFactors.model_validate(
            {
                "diabetes_duration_years": diabetes_duration_years,
                "hba1c_percent": hba1c_percent,
                "fasting_glucose_mg_dl": fasting_glucose_mg_dl,
                "systolic_bp_mmhg": systolic_bp_mmhg,
                "age_years": age_years,
                "kidney_disease": kidney_disease,
            }
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{FIELD_LABELS.get(str(error['loc'][0]), 'Form') if error['loc'] else 'Form'}: "
            f"{error['msg'].removeprefix('Value error, ')}"
            for error in exc.errors()
        )
        raise gr.Error(f"Check these values. {problems}") from exc
    return render_risk(assess(factors, config), _max_points(config))


def _max_points(config: RiskConfig) -> int:
    """Return the highest total the configured bands and kidney points allow.

    Args:
        config: Risk bands and cut-offs.

    Returns:
        The maximum number of points.
    """
    bands = (
        config.diabetes_duration_years,
        config.hba1c_percent,
        config.systolic_bp_mmhg,
        config.age_years,
    )
    return sum(max((band.points for band in factor), default=0) for factor in bands) + (
        config.kidney_disease_points
    )


def build_app(screener: Screener, risk: RiskConfig, audit: AuditLog) -> gr.Blocks:
    """Wire the handlers to a two-tab Gradio interface.

    Start it with ``launch`` so it gets its theme, styles and typeface.

    Args:
        screener: Loaded models.
        risk: Risk bands and cut-offs.
        audit: Audit log for screening outcomes.

    Returns:
        The interface.
    """
    app = gr.Blocks(
        title=TITLE,
        delete_cache=(UPLOAD_CACHE_SECONDS, UPLOAD_CACHE_SECONDS),
        analytics_enabled=False,  # no usage telemetry from a medical screening tool
    )
    with app:
        gr.HTML(MASTHEAD)
        with gr.Tab("Screen a photo"), gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=280):
                photo = gr.Image(
                    label="Fundus photo",
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload"],
                    height=340,
                )
                screen_button = gr.Button("Screen photo", variant="primary", size="lg")
                gr.HTML(PHOTO_TIPS)
            with gr.Column(scale=3, min_width=320):
                result = gr.HTML(EMPTY_SCREEN)
        screen_button.click(
            lambda image: screen_photo(screener, audit, image),
            inputs=photo,
            outputs=result,
        )
        with gr.Tab("Risk calculator"), gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=280):
                duration = gr.Number(label=FIELD_LABELS["diabetes_duration_years"], minimum=0)
                hba1c = gr.Number(label=FIELD_LABELS["hba1c_percent"])
                glucose = gr.Number(
                    label=FIELD_LABELS["fasting_glucose_mg_dl"],
                    info="Used only when HbA1c is left empty.",
                )
                systolic = gr.Number(label=FIELD_LABELS["systolic_bp_mmhg"])
                age = gr.Number(label=FIELD_LABELS["age_years"], minimum=0)
                kidney = gr.Checkbox(label=FIELD_LABELS["kidney_disease"])
                risk_button = gr.Button("Assess priority", variant="primary", size="lg")
            with gr.Column(scale=3, min_width=320):
                risk_result = gr.HTML(EMPTY_RISK)
        risk_button.click(
            lambda *values: assess_risk(risk, *values),
            inputs=[duration, hba1c, glucose, systolic, age, kidney],
            outputs=risk_result,
        )
    return app


def launch(app: gr.Blocks) -> None:
    """Serve an interface from ``build_app`` with its theme, styles and typeface.

    Args:
        app: The interface.
    """
    app.launch(theme=THEME, css=CSS, head=HEAD, footer_links=["settings"])


def create_app(config: AppConfig) -> gr.Blocks:
    """Load models and configs and build the interface.

    Args:
        config: App configuration.

    Returns:
        The interface, ready to ``launch``.

    Raises:
        ConfigError: If the risk config is invalid.
        ModelError: If a checkpoint cannot be obtained or loaded.
    """
    quality, grading = config.checkpoints.resolve()
    screener = Screener.from_checkpoints(quality, grading, config.screener)
    risk = load_config(config.risk_config, RiskConfig)
    logger.info("Audit log: %s", config.audit_log)
    return build_app(screener, risk, AuditLog(config.audit_log))
