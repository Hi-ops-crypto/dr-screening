"""Animated web frontend: one static page plus two JSON endpoints.

The page lives in ``static/index.html``. The endpoints reuse the same
screener, risk calculator and audit log as the Gradio interface, which stays
available at ``/classic``.
"""

import base64
from pathlib import Path
from typing import Annotated, Any

import cv2
import gradio as gr
import numpy as np
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

from drscreen.app.audit import AuditLog
from drscreen.app.ui import FIELD_LABELS, AppConfig, _max_points, build_app
from drscreen.app.views import CSS, HEAD, JPEG_QUALITY, THEME
from drscreen.config import load_config
from drscreen.exceptions import InputError
from drscreen.inference.referral import REFERABLE_MIN_GRADE
from drscreen.inference.screener import Screener
from drscreen.risk.calculator import ClinicalFactors, RiskConfig, assess

PAGE = Path(__file__).with_name("static") / "index.html"
MAX_UPLOAD_BYTES = 40_000_000  # ponytail: whole upload read into memory; stream if limits grow


def _jpeg(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else ""


def screen_bytes(screener: Screener, audit: AuditLog, data: bytes) -> dict[str, Any]:
    """Decode an uploaded file, screen it, record the outcome and return it as JSON.

    Raises:
        HTTPException: 400 if the file is not a usable image.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "The file is over 40 MB. Upload a smaller photo.")
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "That file is not an image. Upload a JPEG or PNG fundus photo.")
    try:
        result = screener.screen(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    except InputError as exc:
        raise HTTPException(400, str(exc)) from exc
    quality, grade = result.quality, result.grade
    audit.record(quality.label, None if grade is None else grade.grade)
    body: dict[str, Any] = {
        "quality": {"label": quality.label, "confidence": quality.confidence},
        "retake": result.retake_message,
        "grade": None,
    }
    if grade is not None:
        body["grade"] = {
            "grade": grade.grade,
            "name": grade.name,
            "probabilities": list(grade.probabilities.items()),
            "referable": grade.referable_probability,
            "referable_min_grade": REFERABLE_MIN_GRADE,
            "refer": grade.referral.refer,
            "guidance": grade.referral.guidance,
            "photo": _jpeg(grade.photo),
            "heatmap": _jpeg(grade.heatmap),
        }
    return body


def risk_json(config: RiskConfig, values: dict[str, Any]) -> dict[str, Any]:
    """Score form values into a tier.

    Raises:
        HTTPException: 422 naming each implausible or missing field.
    """
    try:
        factors = ClinicalFactors.model_validate(values)
    except ValidationError as exc:
        problems = [
            f"{FIELD_LABELS.get(str(e['loc'][0]), 'Form') if e['loc'] else 'Form'}: "
            f"{e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        ]
        raise HTTPException(422, "; ".join(problems)) from exc
    result = assess(factors, config)
    return {
        "tier": result.tier.value,
        "points": result.points,
        "max_points": _max_points(config),
        "factors": result.factors,
        "notes": result.notes,
    }


def build_web(screener: Screener, risk: RiskConfig, audit: AuditLog) -> FastAPI:
    """Serve the animated page at ``/`` and the Gradio interface at ``/classic``."""
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @api.post("/api/screen")
    def screen(file: Annotated[bytes, File()]) -> dict[str, Any]:
        return screen_bytes(screener, audit, file)

    @api.post("/api/risk")
    def risk_endpoint(values: dict[str, Any]) -> dict[str, Any]:
        return risk_json(risk, values)

    gr.mount_gradio_app(
        api, build_app(screener, risk, audit), path="/classic", theme=THEME, css=CSS, head=HEAD
    )
    return api


def create_web(config: AppConfig) -> FastAPI:
    """Load models and configs and build the web app."""
    quality, grading = config.checkpoints.resolve()
    screener = Screener.from_checkpoints(quality, grading, config.screener)
    return build_web(
        screener, load_config(config.risk_config, RiskConfig), AuditLog(config.audit_log)
    )
