"""Evaluate trained checkpoints on held-out data, and tune their referral threshold.

Several checkpoints are evaluated as an ensemble that averages their
probabilities (see ``drscreen.models.ensemble``), optionally with flip
test-time augmentation. The preprocessing recipe and input statistics come
from the checkpoints, not
from current config files, so the evaluation reproduces exactly what the model
was trained on. The cache being evaluated must have been cropped with the
same geometric settings, which is checked before any prediction is made.

Evaluation scores only rows marked external, and threshold tuning only rows
marked validation, so the two never share an image. The external rows are
never used to choose anything, so their metrics stay an honest estimate.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import Field
from torch.utils.data import DataLoader

from drscreen.config import StrictModel
from drscreen.data.dataset import FundusDataset
from drscreen.data.prepare import require_cache_matches
from drscreen.data.schema import CLASS_NAMES, LABEL_COLUMN, SPLIT, Source, Split, validate_manifest
from drscreen.data.settings import DataConfig
from drscreen.devices import DeviceName, resolve_device
from drscreen.evaluation.metrics import (
    BootstrapConfig,
    Labels,
    Probabilities,
    choose_referral_threshold,
    confusion,
    summarise,
    threshold_referral_metrics,
)
from drscreen.evaluation.predict import predict_ensemble
from drscreen.evaluation.report import write_report
from drscreen.exceptions import ConfigError, DataError, ModelError
from drscreen.models.ensemble import Ensemble

logger = logging.getLogger(__name__)

REPORT_DIR_PREFIX = "eval-"
THRESHOLD_FILE = "referral_threshold.json"


class ReferralTuningConfig(StrictModel):
    """Probability-threshold referral rule: how to tune it and which value to report."""

    tuning_sources: list[Source] = Field(min_length=1)
    target_sensitivity: float = Field(gt=0, le=1)
    threshold: float | None = Field(ge=0, le=1)


class EvaluationConfig(StrictModel):
    """Root schema of ``configs/evaluate.yaml``."""

    data_config: Path
    source: Source
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    device: DeviceName
    flip_tta: bool
    referable_min_grade: int | None = Field(ge=0)
    bootstrap: BootstrapConfig
    referral: ReferralTuningConfig


def evaluate_checkpoint(
    checkpoints: Sequence[Path], config: EvaluationConfig, data: DataConfig, output_dir: Path
) -> dict[str, float | None]:
    """Score checkpoints, as an ensemble, on every image of a preprocessed dataset.

    The report is written to ``<output_dir>/eval-<source>/``. When
    ``config.referral.threshold`` is set, the report also covers the
    probability-threshold referral rule at that threshold.

    Args:
        checkpoints: One or more ``.pt`` files written during training.
        config: Evaluation settings.
        data: Data pipeline configuration, for cache locations.
        output_dir: Directory to write the report under.

    Returns:
        The metrics, also saved in the report.

    Raises:
        ModelError: If the checkpoint is unusable or predicts different classes.
        DataError: If the dataset cache is missing or was cropped differently.
    """
    y_true, probabilities, ensemble = _score(
        checkpoints, config, data, [config.source], Split.EXTERNAL
    )
    class_names = ensemble.metadata.class_names
    y_pred: Labels = probabilities.argmax(axis=1)
    metrics = summarise(y_true, y_pred, class_names, config.referable_min_grade, config.bootstrap)
    threshold = config.referral.threshold
    if threshold is not None and config.referable_min_grade is not None:
        metrics |= threshold_referral_metrics(
            y_true, probabilities, config.referable_min_grade, threshold, config.bootstrap
        )
    report_dir = output_dir / f"{REPORT_DIR_PREFIX}{config.source.value}"
    write_report(report_dir, metrics, confusion(y_true, y_pred, len(class_names)), class_names)
    logger.info(
        "%s (flip_tta=%s) on %d %s images: qwk=%s accuracy=%s; report in %s",
        "+".join(ensemble.run_ids),
        config.flip_tta,
        len(y_true),
        config.source.value,
        metrics["qwk"],
        metrics["accuracy/exact"],
        report_dir,
    )
    return metrics


def tune_referral_threshold(
    checkpoints: Sequence[Path], config: EvaluationConfig, data: DataConfig, output_dir: Path
) -> dict[str, object]:
    """Choose the referral threshold on the tuning sources' pooled validation splits.

    The result, which also records the ensemble's validation metrics for
    comparing candidates, is written to ``<output_dir>/referral_threshold.json``.

    Args:
        checkpoints: One or more grading checkpoints, used as an ensemble.
        config: Evaluation settings, including the tuning target.
        data: Data pipeline configuration, for cache locations.
        output_dir: Directory to write the result to.

    Returns:
        The threshold, the grading runs and flip setting it belongs to, and
        validation metrics.

    Raises:
        ConfigError: If the config has no referable grade.
        ModelError: If the checkpoint is unusable or predicts different classes.
        DataError: If the dataset cache is missing or was cropped differently.
    """
    tuning = config.referral
    min_grade = config.referable_min_grade
    if min_grade is None:
        raise ConfigError("referable_min_grade must be set to tune a referral threshold.")
    y_true, probabilities, ensemble = _score(
        checkpoints, config, data, tuning.tuning_sources, Split.VAL
    )
    threshold = choose_referral_threshold(
        y_true, probabilities, min_grade, tuning.target_sensitivity
    )
    validation = summarise(
        y_true, probabilities.argmax(axis=1), ensemble.metadata.class_names, min_grade, None
    ) | threshold_referral_metrics(y_true, probabilities, min_grade, threshold, bootstrap=None)
    result: dict[str, object] = {
        "grading_run_ids": ensemble.run_ids,
        "flip_tta": config.flip_tta,
        "threshold": threshold,
        "tuning_sources": [source.value for source in tuning.tuning_sources],
        "tuning_images": len(y_true),
        "target_sensitivity": tuning.target_sensitivity,
        "validation": validation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / THRESHOLD_FILE
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "%s (flip_tta=%s): validation qwk=%s accuracy=%s; referral threshold %.4f gives "
        "sensitivity %s on %d %s validation images; saved to %s",
        "+".join(ensemble.run_ids),
        config.flip_tta,
        validation["qwk"],
        validation["accuracy/exact"],
        threshold,
        validation["referable@threshold/sensitivity"],
        len(y_true),
        "+".join(source.value for source in tuning.tuning_sources),
        path,
    )
    return result


def _score(
    checkpoints: Sequence[Path],
    config: EvaluationConfig,
    data: DataConfig,
    sources: list[Source],
    split: Split,
) -> tuple[Labels, Probabilities, Ensemble]:
    """Predict ensemble class probabilities for the cached images of one or more datasets.

    Args:
        checkpoints: One or more ``.pt`` files written during training.
        config: Loader, device and test-time augmentation settings.
        data: Data pipeline configuration, for cache locations.
        sources: Datasets to score, all labelled like the checkpoint.
        split: Only rows of this split.

    Returns:
        True labels, averaged softmax probabilities and the loaded ensemble.

    Raises:
        ModelError: If a checkpoint is unusable, the checkpoints are
            incompatible, or they predict different classes.
        DataError: If the dataset cache is missing or was cropped differently.
    """
    device = resolve_device(config.device)
    ensemble = Ensemble.load(checkpoints, device)
    metadata = ensemble.metadata
    manifests = []
    for source in sources:
        label_column = LABEL_COLUMN[source]
        class_names = CLASS_NAMES[label_column]
        if metadata.class_names != class_names:
            raise ModelError(
                f"Checkpoint predicts {metadata.class_names} but {source.value} "
                f"is labelled with {class_names}."
            )
        manifest_path = data.paths.processed_manifest(source)
        require_cache_matches(manifest_path.parent, metadata.preprocessing)
        manifests.append(pd.read_csv(manifest_path))
    manifest = pd.concat(manifests, ignore_index=True)
    validate_manifest(manifest, label_column, len(class_names))
    manifest = manifest[manifest[SPLIT] == split.value]
    if manifest.empty:
        names = "+".join(source.value for source in sources)
        raise DataError(f"{names} has no {split.value} images to score.")

    dataset = FundusDataset(
        manifest, label_column, metadata.preprocessing, mean=metadata.mean, std=metadata.std
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    y_true, probabilities = predict_ensemble(ensemble, loader, device, config.flip_tta)
    return y_true, probabilities, ensemble
