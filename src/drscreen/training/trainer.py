"""Training loop with class weighting, checkpointing and structured logs.

Each run gets its own directory under ``runs_dir`` containing:

* ``run.json``: seed, git commit, library versions, device and resolved config.
* ``metrics.jsonl``: one JSON object per epoch (losses, learning rate,
  validation metrics, duration).
* ``last.pt``: the latest epoch; ``best.pt``: the epoch with the highest
  validation quadratic weighted kappa.

Training stops early when validation kappa has not improved for
``early_stopping_patience`` epochs.
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from drscreen.data.augmentation import build_augmentation
from drscreen.data.dataset import FundusDataset
from drscreen.data.prepare import require_cache_matches
from drscreen.data.preprocessing import PreprocessingConfig
from drscreen.data.schema import CLASS_NAMES, LABEL_COLUMN, SPLIT, Split, validate_manifest
from drscreen.data.settings import DataConfig
from drscreen.devices import resolve_device
from drscreen.evaluation.metrics import summarise
from drscreen.evaluation.predict import predict
from drscreen.exceptions import DataError, ModelError
from drscreen.models.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from drscreen.models.factory import ModelConfig, build_classifier, input_stats
from drscreen.reproducibility import git_state, seed_everything
from drscreen.training.settings import OptimisationConfig, TrainingRunConfig

logger = logging.getLogger(__name__)

RUN_FILE = "run.json"
METRICS_LOG = "metrics.jsonl"
BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
SELECTION_METRIC = "qwk"
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class _Splits:
    """Training and validation rows plus what is needed to interpret them."""

    train: pd.DataFrame
    val: pd.DataFrame
    label_column: str
    class_names: tuple[str, ...]
    preprocessing: PreprocessingConfig


def class_weights(labels: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    """Inverse-frequency class weights, ``(n_samples / (n_classes * n_class)) ** power``.

    With ``power`` 1 every class contributes equally to the loss in
    expectation, which stops the majority class (no DR) from dominating but
    pushes borderline images towards rarer, higher grades. Smaller powers
    soften the correction; 0 gives unweighted cross-entropy.

    Args:
        labels: Integer training labels.
        num_classes: Number of classes.
        power: Exponent applied to the inverse-frequency weights.

    Returns:
        A float32 tensor of one weight per class.

    Raises:
        DataError: If a class has no training examples.
    """
    counts = np.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        raise DataError(f"Classes {missing} have no training images; cannot weight the loss.")
    return torch.tensor((len(labels) / (num_classes * counts)) ** power, dtype=torch.float32)


def run_training(config: TrainingRunConfig, data: DataConfig, repo_dir: Path) -> Path:
    """Train a classifier as described by ``config`` and return its run directory.

    Args:
        config: Training configuration.
        data: Data pipeline configuration, for cache locations and the
            preprocessing recipe.
        repo_dir: Repository root, used to record the git commit.

    Returns:
        The run directory.

    Raises:
        DataError: If the preprocessed cache is missing, stale or lacks a class.
        ModelError: If the backbone cannot be built.
    """
    options = config.optimisation
    seed_everything(options.seed, options.deterministic)
    device = resolve_device(options.device)
    splits = _load_splits(config, data)
    label_column, class_names = splits.label_column, splits.class_names
    preprocessing, train_rows, val_rows = splits.preprocessing, splits.train, splits.val

    model, model_config, init_run_id = _initial_model(config, splits, device)
    mean, std = input_stats(model)
    augmentation = build_augmentation(config.augmentation, preprocessing.image_size)
    train_loader = _loader(
        FundusDataset(
            train_rows, label_column, preprocessing, mean=mean, std=std, augmentation=augmentation
        ),
        options,
        shuffle=True,
        device=device,
    )
    val_loader = _loader(
        FundusDataset(val_rows, label_column, preprocessing, mean=mean, std=std),
        options,
        shuffle=False,
        device=device,
    )
    weights = class_weights(
        train_rows[label_column].to_numpy(), len(class_names), options.class_weight_power
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=options.epochs)
    use_amp = options.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    sources = "+".join(source.value for source in config.sources)
    run_id = f"{sources}-{datetime.now(UTC).strftime(RUN_ID_TIME_FORMAT)}"
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True)
    _write_run_record(
        run_dir,
        {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "seed": options.seed,
            "git": git_state(repo_dir),
            "torch_version": torch.__version__,
            "device": str(device),
            "train_images": len(train_rows),
            "val_images": len(val_rows),
            "class_weights": weights.tolist(),
            "init_run_id": init_run_id,
            "config": config.model_dump(mode="json"),
            "preprocessing": preprocessing.model_dump(mode="json"),
        },
    )
    logger.info(
        "Run %s: %d training and %d validation images", run_id, len(train_rows), len(val_rows)
    )

    best_score = -math.inf
    epochs_without_improvement = 0
    for epoch in range(1, options.epochs + 1):
        started = time.perf_counter()
        learning_rate = scheduler.get_last_lr()[0]
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()
        y_true, probabilities, val_loss = predict(model, val_loader, device)
        y_pred = probabilities.argmax(axis=1)
        metrics = summarise(y_true, y_pred, class_names, config.referable_min_grade, bootstrap=None)
        _log_epoch(
            run_dir,
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.perf_counter() - started, 1),
                **metrics,
            },
        )

        metadata = CheckpointMetadata(
            run_id=run_id,
            epoch=epoch,
            model=model_config,
            class_names=class_names,
            preprocessing=preprocessing,
            mean=mean,
            std=std,
            metrics=metrics,
        )
        save_checkpoint(run_dir / LAST_CHECKPOINT, model, metadata)
        score = metrics[SELECTION_METRIC]
        if score is not None and score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(run_dir / BEST_CHECKPOINT, model, metadata)
            logger.info("New best validation %s %.4f", SELECTION_METRIC, score)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= options.early_stopping_patience:
                logger.info("No improvement for %d epochs; stopping.", epochs_without_improvement)
                break
    if not (run_dir / BEST_CHECKPOINT).is_file():
        save_checkpoint(run_dir / BEST_CHECKPOINT, model, metadata)
        logger.warning(
            "Validation %s was never defined; best.pt is the last epoch.", SELECTION_METRIC
        )
    return run_dir


def _initial_model(
    config: TrainingRunConfig, splits: _Splits, device: torch.device
) -> tuple[torch.nn.Module, ModelConfig, str | None]:
    """Build a fresh classifier, or load the checkpoint to fine-tune from.

    Args:
        config: Training configuration.
        splits: Loaded data, for the class names and preprocessing recipe.
        device: Device to put the model on.

    Returns:
        The model in training mode on ``device``, the model config to record in new
        checkpoints, and the run id it was initialised from (``None`` if fresh).

    Raises:
        ModelError: If the backbone cannot be built, or the starting checkpoint
            has another backbone, other classes or another preprocessing recipe.
    """
    if config.init_checkpoint is None:
        model = build_classifier(config.model, len(splits.class_names))
        return model.to(device), config.model, None
    model, start = load_checkpoint(config.init_checkpoint, device)
    mismatches = {
        "backbone": (start.model.backbone, config.model.backbone),
        "class names": (start.class_names, splits.class_names),
        "preprocessing": (start.preprocessing, splits.preprocessing),
    }
    for name, (found, expected) in mismatches.items():
        if found != expected:
            raise ModelError(
                f"Cannot fine-tune from {config.init_checkpoint}: its {name} {found} "
                f"differs from {expected}."
            )
    logger.info("Fine-tuning from %s (run %s)", config.init_checkpoint, start.run_id)
    return model.train(), start.model, start.run_id


def _load_splits(config: TrainingRunConfig, data: DataConfig) -> _Splits:
    """Read the preprocessed manifests of the configured sources and split them.

    Training and validation rows of every source are pooled, so the best
    checkpoint is the one that does best across all the sources' validation images.

    Args:
        config: Training configuration.
        data: Data pipeline configuration.

    Returns:
        Training and validation rows with their label column, class names and recipe.

    Raises:
        DataError: If the cache is missing, was cropped with other settings,
            or the manifest is invalid.
    """
    label_column = LABEL_COLUMN[config.sources[0]]
    class_names = CLASS_NAMES[label_column]
    preprocessing = data.preprocessing.for_source(config.sources[0])
    manifests = []
    for source in config.sources:
        manifest_path = data.paths.processed_manifest(source)
        if not manifest_path.is_file():
            raise DataError(
                f"No preprocessed manifest at {manifest_path}; run `make prepare-data`."
            )
        require_cache_matches(manifest_path.parent, preprocessing)
        rows = pd.read_csv(manifest_path)
        if not (rows[SPLIT] == Split.TRAIN.value).any():
            raise DataError(
                f"{source.value} has no training images. IDRiD is external test data unless "
                "sources.idrid.adaptation is enabled in the data config."
            )
        manifests.append(rows)
    manifest = pd.concat(manifests, ignore_index=True)
    validate_manifest(manifest, label_column, len(class_names))
    return _Splits(
        train=manifest[manifest[SPLIT] == Split.TRAIN.value],
        val=manifest[manifest[SPLIT] == Split.VAL.value],
        label_column=label_column,
        class_names=class_names,
        preprocessing=preprocessing,
    )


def _log_epoch(run_dir: Path, record: dict[str, object]) -> None:
    """Append one epoch's record to the metrics log and summarise it in the log.

    Args:
        run_dir: Run directory.
        record: JSON-serialisable epoch record including ``epoch`` and ``qwk``.
    """
    with (run_dir / METRICS_LOG).open("a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
    logger.info(
        "Epoch %s: train_loss=%.4f val_loss=%.4f val_%s=%s",
        record["epoch"],
        record["train_loss"],
        record["val_loss"],
        SELECTION_METRIC,
        record[SELECTION_METRIC],
    )


def _loader(
    dataset: FundusDataset, options: OptimisationConfig, shuffle: bool, device: torch.device
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Create a data loader whose shuffling and worker seeds follow the run seed.

    Args:
        dataset: Dataset to load from.
        options: Batch size, worker count and seed.
        shuffle: Whether to shuffle (training) or keep order (validation).
        device: Target device; memory is pinned for CUDA.

    Returns:
        The data loader.
    """
    return DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=shuffle,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
        generator=torch.Generator().manual_seed(options.seed),
    )


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    *,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Run one pass over the training data.

    Args:
        model: Model to train.
        loader: Training loader.
        criterion: Class-weighted loss.
        optimiser: Optimiser.
        scaler: Gradient scaler for mixed precision; a no-op when disabled.
        device: Device the model is on.
        use_amp: Whether to run the forward pass in float16.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_images = 0
    for inputs, labels in loader:
        images, targets = inputs.to(device), labels.to(device)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(targets)
        total_images += len(targets)
    return total_loss / total_images


def _write_run_record(run_dir: Path, record: dict[str, object]) -> None:
    """Write the run's provenance record.

    Args:
        run_dir: Run directory.
        record: JSON-serialisable provenance.
    """
    (run_dir / RUN_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
