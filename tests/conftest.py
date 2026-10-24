"""Shared fixtures: synthetic fundus-like images and tiny checkpoints, so tests need no
real datasets or trained models."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

from drscreen.data.image_io import write_rgb
from drscreen.data.preprocessing import BenGrahamConfig, PreprocessingConfig
from drscreen.models.checkpoint import CheckpointMetadata, save_checkpoint
from drscreen.models.factory import ModelConfig, build_classifier

FundusFactory = Callable[..., npt.NDArray[np.uint8]]

FUNDUS_COLOUR = (190, 90, 40)  # orange-red, like a real retina
TINY_MODEL = ModelConfig(
    backbone="test_efficientnet", pretrained=False, weights_path=None, drop_rate=0.0
)
QUALITY_CLASSES = ("Good", "Usable", "Reject")
GRADE_CLASSES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")

CheckpointWriter = Callable[[tuple[str, ...], int, bool], Path]


def _make_fundus(
    height: int = 300,
    width: int = 400,
    radius: int = 120,
    centre: tuple[int, int] | None = None,
    colour: tuple[int, int, int] = FUNDUS_COLOUR,
) -> npt.NDArray[np.uint8]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = centre if centre is not None else (width // 2, height // 2)
    cv2.circle(image, (cx, cy), radius, colour, thickness=-1)
    return image


@pytest.fixture
def make_fundus() -> FundusFactory:
    """Factory for a black RGB image containing one filled 'retina' circle."""
    return _make_fundus


@pytest.fixture
def write_fundus(make_fundus: FundusFactory) -> Callable[[Path], Path]:
    """Write a default synthetic fundus image to a path and return the path."""

    def write(path: Path) -> Path:
        write_rgb(make_fundus(), path)
        return path

    return write


def _recipe(size: int, normalise: bool) -> PreprocessingConfig:
    ben_graham = BenGrahamConfig(
        sigma_radius_fraction=0.0333,
        contrast_gain=4.0,
        brightness_offset=128,
        edge_erosion_fraction=0.03,
    )
    return PreprocessingConfig(
        image_size=size,
        fundus_intensity_threshold=7,
        min_fundus_fraction=0.1,
        ben_graham=ben_graham if normalise else None,
    )


@pytest.fixture
def write_checkpoint(tmp_path: Path) -> CheckpointWriter:
    """Write a tiny checkpoint whose head always predicts ``favoured``."""

    def write(class_names: tuple[str, ...], favoured: int, normalise: bool) -> Path:
        model = build_classifier(TINY_MODEL, len(class_names)).eval()
        head = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.bias[favoured] = 100.0
        path = tmp_path / f"{len(class_names)}-{favoured}.pt"
        metadata = CheckpointMetadata(
            run_id=path.stem,
            epoch=1,
            model=TINY_MODEL,
            class_names=class_names,
            preprocessing=_recipe(48 if normalise else 32, normalise),
            mean=(0.5, 0.5, 0.5),
            std=(0.25, 0.25, 0.25),
            metrics={},
        )
        save_checkpoint(path, model, metadata)
        return path

    return write
