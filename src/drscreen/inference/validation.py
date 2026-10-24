"""Checks on an uploaded photograph before any model sees it."""

import numpy as np
import numpy.typing as npt
from pydantic import Field

from drscreen.config import StrictModel
from drscreen.data.preprocessing import Image, validate_rgb
from drscreen.exceptions import InputError


class InputLimits(StrictModel):
    """Bounds on the size of an uploaded photograph."""

    min_side: int = Field(gt=0)
    max_pixels: int = Field(gt=0)


def validate_upload(image: npt.NDArray[np.generic] | None, limits: InputLimits) -> Image:
    """Check that an upload is an RGB photograph of a usable size.

    Args:
        image: Decoded upload, or ``None`` when nothing was submitted.
        limits: Size bounds.

    Returns:
        The image, typed as RGB uint8.

    Raises:
        InputError: If there is no image, it is not RGB uint8, or its size is
            outside ``limits``.
    """
    if image is None:
        raise InputError("No image was provided; upload a fundus photograph.")
    try:
        rgb = validate_rgb(image)
    except ValueError as exc:
        raise InputError(f"The upload is not a colour photograph: {exc}") from exc
    height, width = rgb.shape[:2]
    if min(height, width) < limits.min_side:
        raise InputError(
            f"The photograph is {width}x{height} pixels; at least {limits.min_side} pixels "
            "on the shorter side are needed to see retinal detail. Retake it closer or at "
            "a higher resolution."
        )
    if height * width > limits.max_pixels:
        raise InputError(
            f"The photograph is {width}x{height} pixels, more than the "
            f"{limits.max_pixels:,} allowed. Resize it and upload again."
        )
    return rgb
