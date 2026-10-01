from pathlib import Path
from typing import Any

import kagglehub
import pytest
import requests
from kagglehub import exceptions as kaggle_errors

from drscreen.config import load_config
from drscreen.data import fetch
from drscreen.data.settings import DataConfig
from drscreen.exceptions import DataError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shipped_data_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "configs" / "data.yaml", DataConfig)

    assert config.preprocessing.grading.ben_graham is not None
    assert config.preprocessing.quality.ben_graham is None


def test_missing_kaggle_credentials_give_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_credentials(*_: Any, **__: Any) -> str:
        raise kaggle_errors.UnauthenticatedError()

    monkeypatch.setattr(kagglehub, "dataset_download", no_credentials)

    with pytest.raises(DataError, match=r"kaggle\.json"):
        fetch.download_kaggle_dataset("owner/data", tmp_path)


def test_competition_http_error_mentions_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_: Any, **__: Any) -> str:
        raise kaggle_errors.KaggleApiHTTPError("403 Forbidden")

    monkeypatch.setattr(kagglehub, "competition_download", forbidden)

    with pytest.raises(DataError, match="accept the rules"):
        fetch.download_kaggle_competition("some-competition", tmp_path)


class _FailingResponse:
    def __enter__(self) -> "_FailingResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        raise requests.HTTPError("404 Not Found")


def test_failed_download_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(requests, "get", lambda *_, **__: _FailingResponse())
    destination = tmp_path / "labels.csv"

    with pytest.raises(DataError, match="404"):
        fetch.download_file("https://example.org/labels.csv", destination, timeout_seconds=1)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_not_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "labels.csv"
    destination.write_text("cached")
    monkeypatch.setattr(requests, "get", pytest.fail)

    assert fetch.download_file("https://example.org/x.csv", destination, 1) == destination
