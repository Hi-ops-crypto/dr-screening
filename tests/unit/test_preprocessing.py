import numpy as np
import pytest

from drscreen.data.preprocessing import (
    BenGrahamConfig,
    PreprocessingConfig,
    ben_graham_normalise,
    crop_and_resize,
    crop_to_fundus,
    normalise_lighting,
    pad_to_square,
    preprocess,
    resize_square,
)
from drscreen.exceptions import FundusNotFoundError
from tests.conftest import FundusFactory

BEN_GRAHAM = BenGrahamConfig(
    sigma_radius_fraction=1 / 30,
    contrast_gain=4.0,
    brightness_offset=128,
    edge_erosion_fraction=0.01,
)
THRESHOLD = 7


def _config(ben_graham: BenGrahamConfig | None) -> PreprocessingConfig:
    return PreprocessingConfig(
        image_size=64, fundus_intensity_threshold=7, min_fundus_fraction=0.1, ben_graham=ben_graham
    )


def test_crop_returns_bounding_box_of_fundus(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=300, width=400, radius=100, centre=(150, 140))

    cropped = crop_to_fundus(image, intensity_threshold=7, min_fundus_fraction=0.1)

    assert cropped.shape == (201, 201, 3)
    assert cropped[0].max() > 0 and cropped[-1].max() > 0


def test_crop_rejects_black_image() -> None:
    with pytest.raises(FundusNotFoundError, match="locate the retina"):
        crop_to_fundus(np.zeros((100, 100, 3), np.uint8), 7, 0.1)


def test_crop_rejects_image_with_tiny_bright_region(make_fundus: FundusFactory) -> None:
    with pytest.raises(FundusNotFoundError):
        crop_to_fundus(make_fundus(height=400, width=400, radius=20), 7, 0.1)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32)])
def test_rejects_non_rgb_uint8_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        crop_to_fundus(bad, 7, 0.1)


@pytest.mark.parametrize(("height", "width"), [(30, 50), (50, 30), (40, 40)])
def test_pad_to_square_centres_content(height: int, width: int) -> None:
    image = np.full((height, width, 3), 200, np.uint8)

    padded = pad_to_square(image)

    side = max(height, width)
    assert padded.shape == (side, side, 3)
    assert int(padded.sum()) == int(image.sum())
    assert padded[side // 2, side // 2].tolist() == [200, 200, 200]


def test_resize_square_shape() -> None:
    assert resize_square(np.zeros((300, 300, 3), np.uint8), 64).shape == (64, 64, 3)


def test_ben_graham_maps_flat_image_to_offset() -> None:
    flat = np.full((64, 64, 3), 90, np.uint8)

    normalised = ben_graham_normalise(flat, BEN_GRAHAM, THRESHOLD)

    assert normalised.dtype == np.uint8
    assert np.all(np.abs(normalised.astype(int) - 128) <= 1)


def test_ben_graham_removes_global_brightness_difference(make_fundus: FundusFactory) -> None:
    dim = make_fundus(height=128, width=128, radius=64, colour=(60, 30, 15))
    bright = make_fundus(height=128, width=128, radius=64, colour=(200, 100, 50))
    centre = slice(48, 80)

    dim_out = ben_graham_normalise(dim, BEN_GRAHAM, THRESHOLD)[centre, centre]
    bright_out = ben_graham_normalise(bright, BEN_GRAHAM, THRESHOLD)[centre, centre]

    assert np.abs(dim_out.astype(int) - bright_out.astype(int)).max() <= 1


def test_ben_graham_leaves_no_halo_at_retina_edge(make_fundus: FundusFactory) -> None:
    # A uniform retina clipped top and bottom, as some cameras produce.
    image = make_fundus(height=128, width=128, radius=64)
    image[:12] = 0
    image[-12:] = 0

    normalised = ben_graham_normalise(image, BEN_GRAHAM, THRESHOLD)

    assert np.abs(normalised.astype(int) - 128).max() <= 1


def test_ben_graham_preserves_local_detail(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=128, width=128, radius=64)
    image[60:68, 60:68] = (60, 20, 10)  # small dark lesion-like spot

    normalised = ben_graham_normalise(image, BEN_GRAHAM, THRESHOLD)

    assert normalised[64, 64].mean() < 128 - 20
    assert np.abs(normalised[30, 64].astype(int) - 128).max() <= 2


def test_ben_graham_fills_background_with_offset(make_fundus: FundusFactory) -> None:
    image = make_fundus(height=128, width=128, radius=50)

    normalised = ben_graham_normalise(image, BEN_GRAHAM, THRESHOLD)

    assert normalised[0, 0].tolist() == [128, 128, 128]


@pytest.mark.parametrize("ben_graham", [BEN_GRAHAM, None])
def test_preprocess_produces_square_uint8(
    make_fundus: FundusFactory, ben_graham: BenGrahamConfig | None
) -> None:
    result = preprocess(make_fundus(), _config(ben_graham))

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test_preprocess_does_not_modify_input(make_fundus: FundusFactory) -> None:
    image = make_fundus()
    original = image.copy()

    preprocess(image, _config(BEN_GRAHAM))

    np.testing.assert_array_equal(image, original)


def test_preprocess_is_deterministic(make_fundus: FundusFactory) -> None:
    image = make_fundus()

    np.testing.assert_array_equal(
        preprocess(image, _config(BEN_GRAHAM)), preprocess(image, _config(BEN_GRAHAM))
    )


def test_preprocess_equals_lighting_after_geometry(make_fundus: FundusFactory) -> None:
    image = make_fundus()
    config = _config(BEN_GRAHAM)

    np.testing.assert_array_equal(
        preprocess(image, config), normalise_lighting(crop_and_resize(image, config), config)
    )


def test_normalise_lighting_is_identity_without_ben_graham(make_fundus: FundusFactory) -> None:
    image = make_fundus()

    assert normalise_lighting(image, _config(None)) is image
