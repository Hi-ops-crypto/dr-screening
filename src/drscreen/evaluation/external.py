"""Evaluate a trained checkpoint on a held-out dataset.

The preprocessing recipe and input statistics come from the checkpoint, not
from current config files, so the evaluation reproduces exactly what the model
was trained on. The cache being evaluated must have been cropped with the
same geometric settings, which is checked before any prediction is made.
"""

import logging
from pathlib import Path

import pandas as pd
from pydantic import Field
from torch.utils.data import DataLoader

from drscreen.config import StrictModel
from drscreen.data.dataset import FundusDataset
from drscreen.data.prepare import require_cache_matches
from drscreen.data.schema import CLASS_NAMES, LABEL_COLUMN, Source, validate_manifest
from drscreen.data.settings import DataConfig
from drscreen.devices import DeviceName, resolve_device
from drscreen.evaluation.metrics import BootstrapConfig, confusion, summarise
from drscreen.evaluation.predict import predict
from drscreen.evaluation.report import write_report
from drscreen.exceptions import ModelError
from drscreen.models.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

REPORT_DIR_PREFIX = "eval-"


class EvaluationConfig(StrictModel):
    """Root schema of ``configs/evaluate.yaml``."""

    data_config: Path
    source: Source
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    device: DeviceName
    referable_min_grade: int | None = Field(ge=0)
    bootstrap: BootstrapConfig


def evaluate_checkpoint(
    checkpoint: Path, config: EvaluationConfig, data: DataConfig
) -> dict[str, float | None]:
    """Score a checkpoint on every image of a preprocessed dataset.

    The report is written next to the checkpoint in ``eval-<source>/``.

    Args:
        checkpoint: Path to a ``.pt`` file written during training.
        config: Evaluation settings.
        data: Data pipeline configuration, for cache locations.

    Returns:
        The metrics, also saved in the report.

    Raises:
        ModelError: If the checkpoint is unusable or predicts different classes.
        DataError: If the dataset cache is missing or was cropped differently.
    """
    device = resolve_device(config.device)
    model, metadata = load_checkpoint(checkpoint, device)
    label_column = LABEL_COLUMN[config.source]
    class_names = CLASS_NAMES[label_column]
    if metadata.class_names != class_names:
        raise ModelError(
            f"Checkpoint predicts {metadata.class_names} but {config.source.value} "
            f"is labelled with {class_names}."
        )
    manifest_path = data.paths.processed_manifest(config.source)
    require_cache_matches(manifest_path.parent, metadata.preprocessing)
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest, label_column, len(class_names))

    dataset = FundusDataset(
        manifest, label_column, metadata.preprocessing, mean=metadata.mean, std=metadata.std
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    y_true, y_pred, _ = predict(model, loader, device)
    metrics = summarise(y_true, y_pred, class_names, config.referable_min_grade, config.bootstrap)
    output_dir = checkpoint.parent / f"{REPORT_DIR_PREFIX}{config.source.value}"
    write_report(output_dir, metrics, confusion(y_true, y_pred, len(class_names)), class_names)
    logger.info(
        "%s on %d %s images: qwk=%s; report in %s",
        checkpoint.name,
        len(y_true),
        config.source.value,
        metrics["qwk"],
        output_dir,
    )
    return metrics
