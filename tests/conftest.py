"""Shared fixtures: synthetic fundus-like images so tests need no real datasets."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from drscreen.data.image_io import write_rgb

FundusFactory = Callable[..., npt.NDArray[np.uint8]]

FUNDUS_COLOUR = (190, 90, 40)  # orange-red, like a real retina


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
