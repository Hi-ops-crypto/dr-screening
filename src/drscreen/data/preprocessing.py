"""Pure image preprocessing shared by training and inference.

Every function takes an RGB ``uint8`` array of shape ``(H, W, 3)``, returns a
new array and never modifies its input. There is no file I/O here: training
and the web app must apply identical preprocessing, so both import these
functions rather than keeping their own copies.

Lighting normalisation follows B. Graham, "Kaggle Diabetic Retinopathy
Detection competition report", University of Warwick, 2015: subtract a
Gaussian-blurred copy of the image to remove low-frequency lighting variation.
Graham's version blurs across the retina's edge, which leaves bright and dark
halos wherever the retina meets the black background, including the flat cuts
some cameras make at the top and bottom. Halo shape differs between cameras
and would let a model tell datasets apart by their borders. Here the local
mean is computed from retina pixels only (normalised convolution), so the edge
produces no halo, and everything outside the retina is set to a constant.
"""

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field

from drscreen.config import StrictModel
from drscreen.exceptions import FundusNotFoundError

Image = npt.NDArray[np.uint8]

RGB_CHANNELS = 3
UINT8_MAX = 255
MIN_BLUR_WEIGHT = 1e-6  # avoids division by zero far outside the retina; masked anyway


class BenGrahamConfig(StrictModel):
    """Parameters of Ben Graham lighting normalisation."""

    sigma_radius_fraction: float = Field(gt=0)
    contrast_gain: float = Field(gt=0)
    brightness_offset: int = Field(ge=0, le=UINT8_MAX)
    edge_erosion_fraction: float = Field(ge=0, lt=0.5)


class PreprocessingConfig(StrictModel):
    """Full preprocessing recipe for one model input."""

    image_size: int = Field(gt=0)
    fundus_intensity_threshold: int = Field(ge=0, lt=UINT8_MAX)
    min_fundus_fraction: float = Field(gt=0, lt=1)
    ben_graham: BenGrahamConfig | None


def validate_rgb(image: npt.NDArray[np.generic]) -> Image:
    """Check that an array is an RGB uint8 image.

    Args:
        image: Array to check.

    Returns:
        The same array, typed as an RGB uint8 image.

    Raises:
        ValueError: If the array is not ``uint8`` with shape ``(H, W, 3)``.
    """
    if image.dtype != np.uint8 or image.ndim != RGB_CHANNELS or image.shape[2] != RGB_CHANNELS:
        raise ValueError(
            f"Expected an RGB uint8 array of shape (H, W, 3), got dtype {image.dtype} "
            f"and shape {image.shape}."
        )
    return image.astype(np.uint8, copy=False)


def fundus_mask(image: Image, intensity_threshold: int) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of the bright fundus pixels.

    Args:
        image: RGB uint8 image.
        intensity_threshold: Grayscale value above which a pixel is fundus.

    Returns:
        Boolean array of shape ``(H, W)``.
    """
    gray = cv2.cvtColor(validate_rgb(image), cv2.COLOR_RGB2GRAY)
    mask: npt.NDArray[np.bool_] = gray > intensity_threshold
    return mask


def crop_to_fundus(image: Image, intensity_threshold: int, min_fundus_fraction: float) -> Image:
    """Crop an image to the bounding box of its bright fundus region.

    Fundus photographs show a bright circular retina on a near-black
    background. Pixels whose grayscale value exceeds ``intensity_threshold``
    are treated as fundus.

    Args:
        image: RGB uint8 image.
        intensity_threshold: Grayscale value above which a pixel is fundus.
        min_fundus_fraction: Minimum fraction of pixels that must be fundus.

    Returns:
        The cropped image.

    Raises:
        FundusNotFoundError: If too few pixels exceed the threshold, which
            means the image is blank, black or not a fundus photograph.
    """
    mask = fundus_mask(image, intensity_threshold)
    fraction = float(mask.mean())
    if fraction < min_fundus_fraction:
        raise FundusNotFoundError(
            f"Only {fraction:.1%} of the image is brighter than the background threshold; "
            f"at least {min_fundus_fraction:.0%} is needed to locate the retina."
        )
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def pad_to_square(image: Image) -> Image:
    """Pad an image with black borders so that it is square and centred.

    Many fundus cameras clip the top and bottom of the circular retina, so the
    cropped region is wider than it is tall. Padding restores the circle's
    aspect ratio without distorting it.

    Args:
        image: RGB uint8 image.

    Returns:
        A square image whose side equals the longer input side.
    """
    height, width = validate_rgb(image).shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    padded = cv2.copyMakeBorder(
        image,
        top,
        side - height - top,
        left,
        side - width - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return validate_rgb(padded)


def resize_square(image: Image, size: int) -> Image:
    """Resize an image to ``size`` x ``size`` pixels.

    Uses area interpolation, which avoids aliasing when downsampling the
    multi-megapixel originals.

    Args:
        image: RGB uint8 image.
        size: Output side length in pixels.

    Returns:
        The resized image.
    """
    resized = cv2.resize(validate_rgb(image), (size, size), interpolation=cv2.INTER_AREA)
    return validate_rgb(resized)


def ben_graham_normalise(
    image: Image, config: BenGrahamConfig, fundus_intensity_threshold: int
) -> Image:
    """Apply halo-free Ben Graham lighting normalisation to a square fundus image.

    Computes ``gain * (image - local_mean) + offset``, where ``local_mean`` is
    a Gaussian-weighted average over retina pixels only. Pixels outside the
    retina mask, eroded by ``edge_erosion_fraction`` of the image side to drop
    anti-aliased rim pixels, are set to ``offset``.

    Args:
        image: Square RGB uint8 image with the fundus centred.
        config: Normalisation parameters.
        fundus_intensity_threshold: Grayscale value above which a pixel is fundus.

    Returns:
        The normalised image.
    """
    side = validate_rgb(image).shape[0]
    retina = fundus_mask(image, fundus_intensity_threshold).astype(np.uint8)
    erosion = round(side * config.edge_erosion_fraction)
    if erosion > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
        retina = np.asarray(cv2.erode(retina, kernel), dtype=np.uint8)
    weight = retina.astype(np.float32)
    pixels = image.astype(np.float32)
    sigma = side / 2 * config.sigma_radius_fraction
    blurred_pixels = cv2.GaussianBlur(pixels * weight[..., None], ksize=(0, 0), sigmaX=sigma)
    blurred_weight = cv2.GaussianBlur(weight, ksize=(0, 0), sigmaX=sigma)
    local_mean = blurred_pixels / np.maximum(blurred_weight, MIN_BLUR_WEIGHT)[..., None]
    normalised = config.contrast_gain * (pixels - local_mean) + config.brightness_offset
    normalised[retina == 0] = config.brightness_offset
    return np.clip(np.rint(normalised), 0, UINT8_MAX).astype(np.uint8)


def preprocess(image: Image, config: PreprocessingConfig) -> Image:
    """Run the full preprocessing recipe on a raw fundus photograph.

    Args:
        image: Raw RGB uint8 photograph.
        config: The recipe to apply.

    Returns:
        A ``config.image_size`` square RGB uint8 image.

    Raises:
        FundusNotFoundError: If no fundus region can be located.
    """
    cropped = crop_to_fundus(image, config.fundus_intensity_threshold, config.min_fundus_fraction)
    resized = resize_square(pad_to_square(cropped), config.image_size)
    if config.ben_graham is None:
        return resized
    return ben_graham_normalise(resized, config.ben_graham, config.fundus_intensity_threshold)
