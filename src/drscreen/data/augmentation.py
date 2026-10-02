"""Training augmentation that simulates phone-camera fundus capture.

Smartphone fundus photographs (through a clip-on lens) differ from clinic
camera images mainly in focus, exposure, white balance, vignetting and
framing. Each transform below imitates one of these. Augmentation runs on the
cropped, resized image before lighting normalisation, as real degradation
would, using torch's random generator so data-loader workers are seeded
reproducibly.
"""

import torch
from pydantic import Field, model_validator
from torchvision.transforms import v2

from drscreen.config import StrictModel
from drscreen.data.preprocessing import UINT8_MAX, Image
from drscreen.data.tensors import to_chw_tensor, to_hwc_image

SQUARE_ASPECT = (1.0, 1.0)  # keep the circular fundus circular when cropping
KERNEL_SIGMAS = 3  # Gaussian kernel spans +/- 3 sigma, which holds >99% of its weight


class Range(StrictModel):
    """A closed interval sampled uniformly."""

    min: float = Field(ge=0)
    max: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must not exceed max ({self.max}).")
        return self


class AugmentationConfig(StrictModel):
    """Strength and probability of each simulated degradation."""

    crop_scale: Range
    rotation_degrees: float = Field(ge=0, le=180)
    flip_probability: float = Field(ge=0, le=1)
    brightness: float = Field(ge=0)
    contrast: float = Field(ge=0)
    saturation: float = Field(ge=0)
    hue: float = Field(ge=0, le=0.5)
    blur_probability: float = Field(ge=0, le=1)
    blur_sigma: Range
    vignette_probability: float = Field(ge=0, le=1)
    vignette_strength: Range


class RandomVignette(torch.nn.Module):
    """Darken the image towards its corners, as cheap phone lenses do.

    Pixel values are multiplied by ``1 - strength * r**2``, where ``r`` is
    the distance from the centre scaled so that the corners are at 1.
    """

    def __init__(self, probability: float, strength: Range) -> None:
        """Create the transform.

        Args:
            probability: Chance of applying the vignette to an image.
            strength: Range of darkening at the corners, from 0 (none) to 1 (black).
        """
        super().__init__()
        self.probability = probability
        self.strength = strength

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the vignette with probability ``self.probability``.

        Args:
            image: uint8 tensor of shape ``(3, H, W)``.

        Returns:
            The vignetted image, or ``image`` unchanged.
        """
        if torch.rand(1).item() >= self.probability:
            return image
        span = self.strength.max - self.strength.min
        strength = self.strength.min + span * torch.rand(1).item()
        height, width = image.shape[-2:]
        rows = torch.linspace(-1, 1, height).view(-1, 1)
        cols = torch.linspace(-1, 1, width).view(1, -1)
        gain = 1 - strength * (rows**2 + cols**2) / 2
        return (image.float() * gain).round().clamp(0, UINT8_MAX).to(torch.uint8)


def build_augmentation(config: AugmentationConfig, image_size: int) -> v2.Compose:
    """Compose the phone-camera augmentation pipeline.

    Args:
        config: Augmentation parameters.
        image_size: Side length of the cached square images.

    Returns:
        A transform mapping a uint8 ``(3, H, W)`` tensor to one of the same shape.
    """
    blur_kernel = 2 * round(KERNEL_SIGMAS * config.blur_sigma.max) + 1
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                image_size,
                scale=(config.crop_scale.min, config.crop_scale.max),
                ratio=SQUARE_ASPECT,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(config.flip_probability),
            v2.RandomRotation(config.rotation_degrees),
            v2.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
            ),
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        blur_kernel, sigma=(config.blur_sigma.min, config.blur_sigma.max)
                    )
                ],
                p=config.blur_probability,
            ),
            RandomVignette(config.vignette_probability, config.vignette_strength),
        ]
    )


def augment(image: Image, transform: v2.Compose) -> Image:
    """Apply a tensor transform to an RGB array.

    Args:
        image: RGB uint8 array.
        transform: Transform from ``build_augmentation``.

    Returns:
        The augmented RGB uint8 array.
    """
    return to_hwc_image(transform(to_chw_tensor(image)))
