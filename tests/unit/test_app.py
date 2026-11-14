import csv
from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from drscreen.app import ui
from drscreen.app.audit import COLUMNS, AuditLog
from drscreen.app.ui import AppConfig, CheckpointSources, HubSource, assess_risk, screen_photo
from drscreen.app.views import render_screening
from drscreen.config import load_config
from drscreen.exceptions import ConfigError, ModelError
from drscreen.inference.screener import (
    GradingPolicy,
    QualityResult,
    Screener,
    ScreenerConfig,
    ScreeningResult,
)
from drscreen.inference.validation import InputLimits
from drscreen.risk.calculator import RiskConfig
from tests.conftest import GRADE_CLASSES, QUALITY_CLASSES, CheckpointWriter, FundusFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(REPO_ROOT / "configs" / "risk.yaml", RiskConfig)
CONFIG = ScreenerConfig(
    device="cpu",
    limits=InputLimits(min_side=64, max_pixels=1_000_000),
    heatmap_alpha=0.4,
    grading=GradingPolicy(run_ids=["unused"], flip_tta=False, referral_threshold=0.3),
)
SHA = "0123456789abcdef0123456789abcdef01234567"


def _screener(write_checkpoint: CheckpointWriter, quality: int, grade: int) -> Screener:
    grading = write_checkpoint(GRADE_CLASSES, grade, True)
    policy = GradingPolicy(run_ids=[grading.stem], flip_tta=False, referral_threshold=0.3)
    return Screener.from_checkpoints(
        write_checkpoint(QUALITY_CLASSES, quality, False),
        [grading],
        CONFIG.model_copy(update={"grading": policy}),
    )


def _audit_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.reader(file))


def test_graded_photo_is_shown_and_audited(
    tmp_path: Path, write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    audit = AuditLog(tmp_path / "audit" / "log.csv")

    html = screen_photo(_screener(write_checkpoint, 0, 2), audit, make_fundus())

    assert 'class="result result--refer"' in html
    assert "Refer to an ophthalmologist soon</p>" in html
    assert "Most likely grade: <strong>2, Moderate NPDR</strong>" in html
    assert html.count('class="scale__col') == len(GRADE_CLASSES)
    assert html.count("data:image/jpeg;base64,") == 2  # photo and heatmap
    assert 'aria-label="Grade probabilities: grade 0 No DR' in html
    rows = _audit_rows(audit.path)
    assert rows[0] == list(COLUMNS)
    assert rows[1][1:] == ["Good", "2"]


@pytest.mark.parametrize(("grade", "tone"), [(0, "routine"), (1, "watch"), (3, "refer")])
def test_decision_colour_follows_the_guidance(
    tmp_path: Path,
    write_checkpoint: CheckpointWriter,
    make_fundus: FundusFactory,
    grade: int,
    tone: str,
) -> None:
    html = screen_photo(
        _screener(write_checkpoint, 0, grade), AuditLog(tmp_path / "log.csv"), make_fundus()
    )

    assert f'class="result result--{tone}"' in html


def test_rejected_photo_asks_for_retake_and_audits_without_grade(
    tmp_path: Path, write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    audit = AuditLog(tmp_path / "log.csv")
    screener = _screener(write_checkpoint, 2, 0)

    html = screen_photo(screener, audit, make_fundus())
    screen_photo(screener, audit, make_fundus())

    assert "Retake the photo" in html
    assert "data:image" not in html
    rows = _audit_rows(audit.path)
    assert rows[0] == list(COLUMNS)
    assert [row[1:] for row in rows[1:]] == [["Reject", ""], ["Reject", ""]]


def test_rounded_probabilities_are_escaped_in_html(
    tmp_path: Path, write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    html = screen_photo(
        _screener(write_checkpoint, 0, 3), AuditLog(tmp_path / "log.csv"), make_fundus()
    )

    assert "&lt;1%" in html and "&gt;99%" in html
    assert "><1%" not in html


def test_rendered_text_is_escaped() -> None:
    result = ScreeningResult(
        QualityResult("Reject", 0.9, accepted=False), None, "<script>alert(1)</script>"
    )

    html = render_screening(result)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_unusable_upload_is_reported_and_not_audited(
    tmp_path: Path, write_checkpoint: CheckpointWriter
) -> None:
    audit = AuditLog(tmp_path / "log.csv")

    with pytest.raises(gr.Error, match="No image"):
        screen_photo(_screener(write_checkpoint, 0, 0), audit, None)
    assert not audit.path.exists()


def test_risk_form_reports_tier_and_factors() -> None:
    html = assess_risk(RISK, 25, None, 250, 150, 70, True)

    assert 'class="result result--refer"' in html
    assert "High screening priority" in html
    assert "7 of 7 points." in html
    assert "<li>Kidney disease<span>+1</span></li>" in html
    assert "estimated" in html


def test_risk_form_names_fields_as_the_form_does() -> None:
    with pytest.raises(gr.Error, match=r"Systolic blood pressure \(mmHg\)"):
        assess_risk(RISK, 5, 7.0, None, None, 40, False)
    with pytest.raises(gr.Error, match="Enter HbA1c or fasting glucose"):
        assess_risk(RISK, 5, None, None, 120, 40, False)


def test_interface_builds(tmp_path: Path, write_checkpoint: CheckpointWriter) -> None:
    app = ui.build_app(_screener(write_checkpoint, 0, 0), RISK, AuditLog(tmp_path / "log.csv"))

    assert isinstance(app, gr.Blocks)


def test_shipped_app_config_uses_local_checkpoints() -> None:
    config = load_config(REPO_ROOT / "configs" / "app.yaml", AppConfig)

    assert config.checkpoints.hub is None
    quality, grading = config.checkpoints.resolve()
    assert quality == Path("models/quality.pt")
    assert grading == [Path(name) for name in config.checkpoints.grading]
    assert len(config.screener.grading.run_ids) == len(grading)


def test_space_config_needs_a_pinned_commit() -> None:
    space_config = REPO_ROOT / "deploy" / "hf_space" / "configs" / "app.yaml"

    with pytest.raises(ConfigError, match="revision"):
        load_config(space_config, AppConfig, ["checkpoints.hub.revision=main"])
    hub = load_config(space_config, AppConfig).checkpoints.hub
    assert hub is not None
    assert len(hub.revision) == len(SHA)


def test_hub_checkpoints_are_downloaded_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_download(repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return f"/cache/{filename}"

    monkeypatch.setattr(ui, "hf_hub_download", fake_download)
    sources = CheckpointSources(
        hub=HubSource(repo_id="user/models", revision=SHA),
        quality="q.pt",
        grading=["g1.pt", "g2.pt"],
    )

    assert sources.resolve() == (Path("/cache/q.pt"), [Path("/cache/g1.pt"), Path("/cache/g2.pt")])
    assert [name for _, name, _ in calls] == ["q.pt", "g1.pt", "g2.pt"]
    assert all(repo == "user/models" and revision == SHA for repo, _, revision in calls)


def test_failed_hub_download_is_a_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_download(repo_id: str, filename: str, revision: str) -> str:
        raise OSError("offline")

    monkeypatch.setattr(ui, "hf_hub_download", failing_download)
    sources = CheckpointSources(
        hub=HubSource(repo_id="user/models", revision=SHA), quality="q.pt", grading=["g.pt"]
    )

    with pytest.raises(ModelError, match="offline"):
        sources.resolve()


def test_audit_log_never_stores_images(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "log.csv")

    audit.record("Good", 0)

    header, row = _audit_rows(audit.path)
    assert header == list(COLUMNS)
    assert len(row) == len(COLUMNS)
    assert np.datetime64(row[0].replace("+00:00", ""))


def test_web_endpoints_serve_page_risk_and_screening(
    tmp_path: Path, write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    import cv2  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from drscreen.app.web import build_web  # noqa: PLC0415

    audit = AuditLog(tmp_path / "log.csv")
    client = TestClient(build_web(_screener(write_checkpoint, 0, 2), RISK, audit))
    assert "Screen a photo" in client.get("/").text

    risk = client.post(
        "/api/risk",
        json={
            "diabetes_duration_years": 25,
            "hba1c_percent": 9.5,
            "fasting_glucose_mg_dl": None,
            "systolic_bp_mmhg": 150,
            "age_years": 64,
            "kidney_disease": True,
        },
    ).json()
    assert (risk["tier"], risk["points"], risk["max_points"]) == ("high", 7, 7)
    assert client.post("/api/risk", json={"age_years": 5}).status_code == 422

    assert client.post("/api/screen", files={"file": b"not an image"}).status_code == 400
    _, png = cv2.imencode(".png", cv2.cvtColor(make_fundus(), cv2.COLOR_RGB2BGR))
    body = client.post("/api/screen", files={"file": png.tobytes()}).json()
    assert body["grade"]["grade"] == 2
    assert body["grade"]["refer"] is True
    assert body["grade"]["heatmap"].startswith("data:image/jpeg;base64,")
    assert _audit_rows(audit.path)[1][2] == "2"
