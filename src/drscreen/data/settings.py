"""Typed schema for ``configs/data.yaml``."""

from pathlib import Path

from pydantic import Field, HttpUrl

from drscreen.config import StrictModel
from drscreen.data.preprocessing import PreprocessingConfig
from drscreen.data.schema import GRADE, LABEL_COLUMN, Source

MANIFEST_FILE = "manifest.csv"


class DataPaths(StrictModel):
    """Where each stage of the data pipeline reads and writes."""

    raw_dir: Path
    manifests_dir: Path
    processed_dir: Path
    reports_dir: Path

    def manifest(self, source: Source) -> Path:
        """Return the path of a source's raw manifest.

        Args:
            source: Dataset.

        Returns:
            Path of the manifest CSV.
        """
        return self.manifests_dir / f"{source.value}.csv"

    def processed_manifest(self, source: Source) -> Path:
        """Return the path of a source's preprocessed-image manifest.

        Args:
            source: Dataset.

        Returns:
            Path of the manifest CSV.
        """
        return self.processed_dir / source.value / MANIFEST_FILE


class AptosSource(StrictModel):
    """APTOS 2019, downloaded from its Kaggle competition."""

    enabled: bool
    kaggle_competition: str


class IdridSource(StrictModel):
    """IDRiD, downloaded from a Kaggle mirror pinned to a version."""

    enabled: bool
    kaggle_dataset: str


class EyeqSource(StrictModel):
    """EyeQ quality labels from GitHub, joined to EyePACS images from Kaggle."""

    enabled: bool
    kaggle_dataset: str
    images_subdir: Path
    label_urls: list[HttpUrl] = Field(min_length=1)


class SourcesConfig(StrictModel):
    """Configuration for every dataset."""

    aptos: AptosSource
    idrid: IdridSource
    eyeq: EyeqSource

    def is_enabled(self, source: Source) -> bool:
        """Return whether a source is enabled.

        Args:
            source: Dataset.

        Returns:
            The source's ``enabled`` flag.
        """
        enabled: bool = getattr(self, source.value).enabled
        return enabled


class SplitConfig(StrictModel):
    """Train/validation split parameters."""

    seed: int
    val_fraction: float = Field(gt=0, le=0.5)


class PreprocessingProfiles(StrictModel):
    """Preprocessing recipe for each model's input."""

    grading: PreprocessingConfig
    quality: PreprocessingConfig

    def for_source(self, source: Source) -> PreprocessingConfig:
        """Return the recipe for the model a dataset trains or evaluates.

        Args:
            source: Dataset.

        Returns:
            The grading recipe for DR-graded datasets, otherwise the quality recipe.
        """
        return self.grading if LABEL_COLUMN[source] == GRADE else self.quality


class PrepareConfig(StrictModel):
    """Batch preprocessing parameters."""

    workers: int = Field(gt=0)


class InspectionConfig(StrictModel):
    """Dataset inspection report parameters."""

    samples_per_class: int = Field(gt=0)
    thumbnail_size: int = Field(gt=0)
    seed: int


class DataConfig(StrictModel):
    """Root schema of ``configs/data.yaml``."""

    paths: DataPaths
    sources: SourcesConfig
    download_timeout_seconds: float = Field(gt=0)
    split: SplitConfig
    preprocessing: PreprocessingProfiles
    prepare: PrepareConfig
    inspection: InspectionConfig
