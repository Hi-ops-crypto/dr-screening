"""Conversion between RGB image arrays and model input tensors.

Training and inference both build model inputs with ``image_to_tensor`` so the
two can never disagree on channel order or normalisation statistics.
"""

import numpy as np
import torch

from drscreen.data.preprocessing import UINT8_MAX, Image, validate_rgb

ChannelStats = tuple[float, float, float]


def to_chw_tensor(image: Image) -> torch.Tensor:
    """Convert an HWC uint8 array to a CHW uint8 tensor without rescaling.

    Args:
        image: RGB uint8 array.

    Returns:
        A uint8 tensor of shape ``(3, H, W)``.
    """
    return torch.from_numpy(np.ascontiguousarray(validate_rgb(image))).permute(2, 0, 1)


def to_hwc_image(tensor: torch.Tensor) -> Image:
    """Convert a CHW uint8 tensor back to an HWC uint8 array.

    Args:
        tensor: A uint8 tensor of shape ``(3, H, W)``.

    Returns:
        RGB uint8 array.
    """
    return validate_rgb(tensor.permute(1, 2, 0).contiguous().numpy())


def image_to_tensor(image: Image, mean: ChannelStats, std: ChannelStats) -> torch.Tensor:
    """Turn a preprocessed RGB image into a normalised float model input.

    Args:
        image: Preprocessed RGB uint8 array.
        mean: Per-channel mean on the ``[0, 1]`` scale the backbone was trained with.
        std: Per-channel standard deviation on the same scale.

    Returns:
        A float32 tensor of shape ``(3, H, W)``.
    """
    scaled = to_chw_tensor(image).float() / UINT8_MAX
    return (scaled - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(std).view(-1, 1, 1)
