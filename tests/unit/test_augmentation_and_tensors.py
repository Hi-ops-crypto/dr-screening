import numpy as np
import pytest
import torch
from pydantic import ValidationError

from drscreen.data.augmentation import (
    AugmentationConfig,
    RandomGaussianBlur,
    RandomVignette,
    Range,
    augment,
    build_augmentation,
)
from drscreen.data.tensors import image_to_tensor, to_chw_tensor, to_hwc_image
from tests.conftest import FundusFactory

CONFIG = AugmentationConfig(
    crop_scale=Range(min=0.8, max=1.0),
    rotation_degrees=20,
    flip_probability=0.5,
    brightness=0.3,
    contrast=0.3,
    saturation=0.2,
    hue=0.03,
    blur_probability=0.5,
    blur_sigma=Range(min=0.5, max=2.0),
    vignette_probability=0.5,
    vignette_strength=Range(min=0.2, max=0.6),
)


def test_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        Range(min=2, max=1)


def test_augmentation_preserves_shape_and_dtype(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=64, width=64, radius=30)

    augmented = augment(image, build_augmentation(CONFIG, image_size=64))

    assert augmented.shape == (64, 64, 3)
    assert augmented.dtype == np.uint8


def test_augmentation_is_reproducible_under_torch_seed(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=64, width=64, radius=30)
    transform = build_augmentation(CONFIG, image_size=64)

    torch.manual_seed(7)
    first = augment(image, transform)
    torch.manual_seed(7)
    second = augment(image, transform)

    np.testing.assert_array_equal(first, second)


def test_vignette_darkens_corners_not_centre() -> None:
    image = torch.full((3, 33, 33), 200, dtype=torch.uint8)

    vignetted = RandomVignette(probability=1.0, strength=Range(min=0.5, max=0.5))(image)

    assert vignetted[0, 16, 16] == 200
    assert vignetted[0, 0, 0] == 100


def test_vignette_with_zero_probability_is_identity() -> None:
    image = torch.full((3, 8, 8), 200, dtype=torch.uint8)

    assert RandomVignette(0.0, Range(min=0.5, max=0.5))(image) is image


def test_tensor_round_trip(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=20, width=30, radius=8)

    np.testing.assert_array_equal(to_hwc_image(to_chw_tensor(image)), image)


def test_image_to_tensor_normalises_per_channel() -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    image[..., 0] = 255

    tensor = image_to_tensor(image, mean=(0.5, 0.0, 0.0), std=(0.5, 1.0, 1.0))

    assert tensor.shape == (3, 2, 2)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[0], torch.ones(2, 2))
    assert torch.allclose(tensor[1], torch.zeros(2, 2))


def test_gaussian_blur_applies_with_its_probability() -> None:
    image = torch.zeros(3, 32, 32, dtype=torch.uint8)
    image[:, 16, 16] = 255

    blurred = RandomGaussianBlur(1.0, Range(min=2.0, max=2.0))(image)

    assert blurred.shape == image.shape and blurred.dtype == torch.uint8
    assert 0 < blurred[0, 16, 16] < 255
    assert blurred[0, 16, 18] > 0
    assert torch.equal(RandomGaussianBlur(0.0, Range(min=2.0, max=2.0))(image), image)
