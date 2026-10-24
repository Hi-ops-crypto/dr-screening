"""Typed schema for training configs (``configs/train_*.yaml``)."""

from pathlib import Path

from pydantic import Field, field_validator

from drscreen.config import StrictModel
from drscreen.data.augmentation import AugmentationConfig
from drscreen.data.schema import LABEL_COLUMN, Source
from drscreen.devices import DeviceName
from drscreen.models.factory import ModelConfig


class OptimisationConfig(StrictModel):
    """Optimiser, schedule, stopping and loader settings."""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    class_weight_power: float = Field(ge=0, le=1)
    early_stopping_patience: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    mixed_precision: bool
    device: DeviceName
    seed: int
    deterministic: bool


class TrainingRunConfig(StrictModel):
    """Root schema of a training config."""

    data_config: Path
    sources: list[Source] = Field(min_length=1)
    runs_dir: Path
    # Fine-tune from this checkpoint instead of the backbone's pretrained
    # weights; ``model.backbone`` must name the same architecture.
    init_checkpoint: Path | None
    model: ModelConfig
    augmentation: AugmentationConfig
    optimisation: OptimisationConfig
    referable_min_grade: int | None = Field(ge=0)

    @field_validator("sources")
    @classmethod
    def _trainable(cls, sources: list[Source]) -> list[Source]:
        if len({LABEL_COLUMN[source] for source in sources}) > 1:
            raise ValueError(f"Sources {sources} are labelled differently; train them separately.")
        if len(set(sources)) != len(sources):
            raise ValueError(f"Sources {sources} contain duplicates.")
        return sources
