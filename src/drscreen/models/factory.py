"""Build image classifiers from timm backbones selected by name.

Swapping the backbone is a config change: any timm model name works, and a
local weights file (for example a RETFound checkpoint converted to a timm
state dict) can replace the default pretrained weights.
"""

from pathlib import Path
from typing import Any

import timm
import torch
from pydantic import Field

from drscreen.config import StrictModel
from drscreen.data.tensors import ChannelStats
from drscreen.exceptions import ModelError


class ModelConfig(StrictModel):
    """Which backbone to build and how to initialise it."""

    backbone: str
    pretrained: bool
    weights_path: Path | None
    drop_rate: float = Field(ge=0, lt=1)


def build_classifier(config: ModelConfig, num_classes: int) -> torch.nn.Module:
    """Create a classifier with a freshly initialised ``num_classes`` head.

    Args:
        config: Backbone selection and initialisation.
        num_classes: Number of output classes.

    Returns:
        The model, in training mode.

    Raises:
        ModelError: If the backbone name is unknown or the weights file is missing.
    """
    if config.backbone not in timm.list_models():
        raise ModelError(
            f"Unknown backbone {config.backbone!r}; see timm.list_models() for valid names."
        )
    overlay = None
    if config.weights_path is not None:
        if not config.weights_path.is_file():
            raise ModelError(f"Backbone weights file not found: {config.weights_path}")
        overlay = {"file": str(config.weights_path)}
    model: torch.nn.Module = timm.create_model(
        config.backbone,
        pretrained=config.pretrained or overlay is not None,
        pretrained_cfg_overlay=overlay,
        num_classes=num_classes,
        drop_rate=config.drop_rate,
    )
    return model


def input_stats(model: torch.nn.Module) -> tuple[ChannelStats, ChannelStats]:
    """Return the per-channel input mean and standard deviation a model expects.

    Args:
        model: A model created by ``build_classifier``.

    Returns:
        ``(mean, std)``, each with one value per RGB channel.

    Raises:
        ModelError: If the model does not carry timm's input configuration.
    """
    pretrained_cfg: dict[str, Any] = getattr(model, "pretrained_cfg", {})
    if "mean" not in pretrained_cfg or "std" not in pretrained_cfg:
        raise ModelError(f"{type(model).__name__} does not declare its input mean and std.")
    mean = [float(value) for value in pretrained_cfg["mean"]]
    std = [float(value) for value in pretrained_cfg["std"]]
    return (mean[0], mean[1], mean[2]), (std[0], std[1], std[2])
