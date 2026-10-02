"""Save and load models together with everything needed to use them.

A checkpoint stores the weights plus metadata: backbone, class names,
preprocessing recipe and input statistics. Whoever loads a checkpoint (the
evaluation script, the web app) takes these from the checkpoint rather than
from current config files, so a model is always fed exactly the inputs it was
trained on.

Metadata is stored as a JSON string so checkpoints load with
``torch.load(weights_only=True)``, which cannot execute code from the file.
"""

import json
import pickle
from pathlib import Path

import torch

from drscreen.config import StrictModel
from drscreen.data.preprocessing import PreprocessingConfig
from drscreen.data.tensors import ChannelStats
from drscreen.exceptions import ModelError
from drscreen.models.factory import ModelConfig, build_classifier

WEIGHTS_KEY = "state_dict"
METADATA_KEY = "metadata"


class CheckpointMetadata(StrictModel):
    """Everything besides weights that is needed to use a trained model."""

    run_id: str
    epoch: int
    model: ModelConfig
    class_names: tuple[str, ...]
    preprocessing: PreprocessingConfig
    mean: ChannelStats
    std: ChannelStats
    metrics: dict[str, float | None]


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: CheckpointMetadata) -> None:
    """Write a checkpoint atomically.

    The file is written under a temporary name and renamed, so an interrupted
    save never leaves a truncated checkpoint behind.

    Args:
        path: Destination file.
        model: Model whose weights are saved.
        metadata: Accompanying metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save({WEIGHTS_KEY: model.state_dict(), METADATA_KEY: metadata.model_dump_json()}, partial)
    partial.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, CheckpointMetadata]:
    """Rebuild a model from a checkpoint, ready for inference.

    Args:
        path: Checkpoint written by ``save_checkpoint``.
        device: Device to load the model onto.

    Returns:
        The model in evaluation mode, and its metadata.

    Raises:
        ModelError: If the file is missing, unreadable, or does not match the
            architecture its metadata describes.
    """
    if not path.is_file():
        raise ModelError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
        metadata = CheckpointMetadata.model_validate(json.loads(payload[METADATA_KEY]))
    except (pickle.UnpicklingError, EOFError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        raise ModelError(
            f"Checkpoint {path} is corrupt or not a drscreen checkpoint: {exc}"
        ) from exc
    untrained = metadata.model.model_copy(update={"pretrained": False, "weights_path": None})
    model = build_classifier(untrained, num_classes=len(metadata.class_names))
    try:
        model.load_state_dict(payload[WEIGHTS_KEY])
    except RuntimeError as exc:
        raise ModelError(
            f"Weights in {path} do not fit backbone {metadata.model.backbone!r}: {exc}"
        ) from exc
    return model.to(device).eval(), metadata
