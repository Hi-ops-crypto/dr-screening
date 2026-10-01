from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drscreen.data.image_io import read_rgb, write_rgb
from drscreen.data.prepare import ERROR_COLUMN, prepare_images
from drscreen.data.preprocessing import PreprocessingConfig
from drscreen.data.schema import GRADE, GROUP, IMAGE_ID, IMAGE_PATH, SOURCE
from drscreen.exceptions import DataError

CONFIG = PreprocessingConfig(
    image_size=32, fundus_intensity_threshold=7, min_fundus_fraction=0.1, ben_graham=None
)


def _manifest(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            IMAGE_ID: [f"img{i}" for i in range(len(paths))],
            IMAGE_PATH: [str(path) for path in paths],
            GRADE: [0] * len(paths),
            SOURCE: "test",
            GROUP: [f"img{i}" for i in range(len(paths))],
        }
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_prepares_images_and_reports_failures(
    tmp_path: Path, write_fundus: Callable[[Path], Path], workers: int
) -> None:
    good = write_fundus(tmp_path / "raw" / "good.png")
    black = tmp_path / "raw" / "black.png"
    write_rgb(np.zeros((50, 50, 3), np.uint8), black)
    missing = tmp_path / "raw" / "missing.png"

    result = prepare_images(_manifest([good, black, missing]), tmp_path / "out", CONFIG, workers)

    assert result.manifest[IMAGE_ID].tolist() == ["img0"]
    assert read_rgb(Path(result.manifest[IMAGE_PATH].iloc[0])).shape == (32, 32, 3)
    assert result.failures[IMAGE_ID].tolist() == ["img1", "img2"]
    assert "retina" in result.failures[ERROR_COLUMN].iloc[0]
    assert "not found" in result.failures[ERROR_COLUMN].iloc[1]


def test_rerun_with_same_recipe_resumes(
    tmp_path: Path, write_fundus: Callable[[Path], Path]
) -> None:
    manifest = _manifest([write_fundus(tmp_path / "a.png")])
    first = prepare_images(manifest, tmp_path / "out", CONFIG, workers=1)
    Path(manifest[IMAGE_PATH].iloc[0]).unlink()  # cached output must be reused

    second = prepare_images(manifest, tmp_path / "out", CONFIG, workers=1)

    assert second.manifest.equals(first.manifest)
    assert second.failures.empty


def test_rerun_with_different_recipe_is_rejected(
    tmp_path: Path, write_fundus: Callable[[Path], Path]
) -> None:
    manifest = _manifest([write_fundus(tmp_path / "a.png")])
    prepare_images(manifest, tmp_path / "out", CONFIG, workers=1)
    changed = CONFIG.model_copy(update={"image_size": 64})

    with pytest.raises(DataError, match="different preprocessing recipe"):
        prepare_images(manifest, tmp_path / "out", changed, workers=1)
