"""Selection of the torch device from a config value."""

import logging
from typing import Literal

import torch

from drscreen.exceptions import ConfigError

logger = logging.getLogger(__name__)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(name: DeviceName) -> torch.device:
    """Turn a configured device name into a torch device.

    Args:
        name: ``"auto"`` picks CUDA, then Apple MPS, then CPU, whichever is
            available first; other names are used as given.

    Returns:
        The selected device.

    Raises:
        ConfigError: If an explicitly requested accelerator is unavailable.
    """
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if name == "auto":
        name = next(device for device in ("cuda", "mps", "cpu") if available[device])
    elif not available[name]:
        raise ConfigError(f"Device {name!r} was requested but is not available on this machine.")
    logger.info("Using device %s", name)
    return torch.device(name)
