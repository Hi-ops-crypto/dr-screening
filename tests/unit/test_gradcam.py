import numpy as np
import pytest
import torch

from drscreen.explain.gradcam import gradcam, last_conv_layer, overlay
from drscreen.models.factory import ModelConfig, build_classifier


def _brightness_classifier() -> torch.nn.Module:
    """Class 1 scores the mean of a 1x1 convolution that copies the red channel."""
    conv = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
    head = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        head.weight.copy_(torch.tensor([[-1.0], [1.0]]))
    return torch.nn.Sequential(conv, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), head).eval()


def test_heatmap_highlights_the_region_that_raised_the_score() -> None:
    model = _brightness_classifier()
    inputs = torch.zeros(1, 3, 32, 32)
    inputs[0, 0, :8, -8:] = 1.0  # bright top-right corner

    result = gradcam(model, inputs, last_conv_layer(model))

    assert result.target_class == 1
    assert result.heatmap.shape == (32, 32)
    assert result.heatmap.max() == pytest.approx(1.0)
    assert result.heatmap[:8, -8:].mean() > 0.9
    assert result.heatmap[-8:, :8].max() < 0.1


def test_explaining_a_class_with_no_support_gives_an_empty_map() -> None:
    model = _brightness_classifier()
    inputs = torch.zeros(1, 3, 16, 16)
    inputs[0, 0, :4, :4] = 1.0

    result = gradcam(model, inputs, last_conv_layer(model), target_class=0)

    assert result.target_class == 0
    assert not result.heatmap.any()


def test_gradcam_leaves_no_hooks_or_parameter_gradients() -> None:
    config = ModelConfig(
        backbone="test_efficientnet", pretrained=False, weights_path=None, drop_rate=0.0
    )
    model = build_classifier(config, num_classes=5).eval()
    layer = last_conv_layer(model)

    result = gradcam(model, torch.rand(1, 3, 64, 64), layer)

    assert result.logits.shape == (5,)
    assert result.heatmap.shape == (64, 64)
    assert 0 <= result.heatmap.min() and result.heatmap.max() <= 1
    assert not layer._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_model_without_convolutions_is_rejected() -> None:
    with pytest.raises(ValueError, match="no Conv2d"):
        last_conv_layer(torch.nn.Linear(3, 2))


def test_overlay_resizes_heatmap_and_keeps_image_format() -> None:
    image = np.full((40, 60, 3), 100, dtype=np.uint8)
    heatmap = np.zeros((10, 10), dtype=np.float32)

    blended = overlay(image, heatmap, alpha=0.0)

    assert blended.dtype == np.uint8
    assert np.array_equal(blended, image)
    assert overlay(image, heatmap + 1, alpha=1.0)[0, 0].tolist() == [128, 0, 0]  # jet at 1.0
    assert np.array_equal(overlay(image, heatmap, alpha=1.0), image)  # no heat, no colour
