"""Typed schema for training configs (``configs/train_*.yaml``)."""

from pathlib import Path

from pydantic import Field, field_validator

from drscreen.config import StrictModel
from drscreen.data.augmentation import AugmentationConfig
from drscreen.data.schema import Source
from drscreen.devices import DeviceName
from drscreen.models.factory import ModelConfig


class OptimisationConfig(StrictModel):
    """Optimiser, schedule, stopping and loader settings."""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    early_stopping_patience: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    mixed_precision: bool
    device: DeviceName
    seed: int
    deterministic: bool


class TrainingRunConfig(StrictModel):
    """Root schema of a training config."""

    data_config: Path
    source: Source
    runs_dir: Path
    model: ModelConfig
    augmentation: AugmentationConfig
    optimisation: OptimisationConfig
    referable_min_grade: int | None = Field(ge=0)

    @field_validator("source")
    @classmethod
    def _not_external(cls, source: Source) -> Source:
        if source is Source.IDRID:
            raise ValueError("IDRiD is the external test set and must never be used for training.")
        return source
