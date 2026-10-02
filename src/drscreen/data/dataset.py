"""Torch dataset over a manifest of cached, cropped fundus images."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from drscreen.data.augmentation import augment
from drscreen.data.image_io import read_rgb
from drscreen.data.preprocessing import PreprocessingConfig, normalise_lighting
from drscreen.data.schema import IMAGE_PATH
from drscreen.data.tensors import ChannelStats, image_to_tensor


class FundusDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load cached images, augment them, normalise lighting and return tensors.

    Each item goes through the same steps as a photograph at inference time,
    with augmentation inserted between cropping (done in the cache) and
    lighting normalisation.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        label_column: str,
        preprocessing: PreprocessingConfig,
        *,
        mean: ChannelStats,
        std: ChannelStats,
        augmentation: v2.Compose | None = None,
    ) -> None:
        """Create the dataset.

        Args:
            manifest: Rows of cached images with ``image_path`` and the label column.
            label_column: Column holding integer class labels.
            preprocessing: Recipe whose lighting step is applied at load time.
            mean: Per-channel input mean expected by the backbone.
            std: Per-channel input standard deviation expected by the backbone.
            augmentation: Optional transform from ``build_augmentation``.
        """
        self.paths = [Path(path) for path in manifest[IMAGE_PATH]]
        self.labels = manifest[label_column].astype(int).tolist()
        self.preprocessing = preprocessing
        self.mean = mean
        self.std = std
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one image and its label.

        Args:
            index: Row index.

        Returns:
            The model input tensor and the integer label.

        Raises:
            DataError: If the cached image cannot be read.
        """
        image = read_rgb(self.paths[index])
        if self.augmentation is not None:
            image = augment(image, self.augmentation)
        image = normalise_lighting(image, self.preprocessing)
        return image_to_tensor(image, self.mean, self.std), self.labels[index]
