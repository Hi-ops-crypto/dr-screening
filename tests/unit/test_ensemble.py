from pathlib import Path

import pytest
import torch

from drscreen.exceptions import ModelError
from drscreen.models.checkpoint import load_checkpoint
from drscreen.models.ensemble import Ensemble
from tests.conftest import GRADE_CLASSES, QUALITY_CLASSES, CheckpointWriter


def _flip_sensitive_model() -> torch.nn.Module:
    """Scores class 1 by the mean of the left half of the image, class 0 by the right."""

    class Halves(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            half = inputs.shape[-1] // 2
            left = inputs[..., :half].mean(dim=(1, 2, 3))
            right = inputs[..., half:].mean(dim=(1, 2, 3))
            return torch.stack([right, left], dim=1) * 10

    return Halves().eval()


def test_single_member_without_flip_is_plain_softmax(write_checkpoint: CheckpointWriter) -> None:
    path = write_checkpoint(GRADE_CLASSES, 2, True)
    model, _ = load_checkpoint(path, torch.device("cpu"))
    inputs = torch.rand(2, 3, 48, 48)

    probabilities = Ensemble.load([path], torch.device("cpu")).probabilities(inputs, flip=False)

    assert torch.allclose(probabilities, torch.softmax(model(inputs), dim=1))


def test_members_are_averaged(write_checkpoint: CheckpointWriter) -> None:
    paths = [write_checkpoint(GRADE_CLASSES, 0, True), write_checkpoint(GRADE_CLASSES, 3, True)]
    ensemble = Ensemble.load(paths, torch.device("cpu"))

    probabilities = ensemble.probabilities(torch.rand(1, 3, 48, 48), flip=False)

    assert ensemble.run_ids == [path.stem for path in paths]
    assert probabilities[0].tolist() == pytest.approx([0.5, 0, 0, 0.5, 0], abs=1e-6)


def test_flip_averages_each_image_with_its_mirror(write_checkpoint: CheckpointWriter) -> None:
    _, metadata = load_checkpoint(write_checkpoint(GRADE_CLASSES, 0, True), torch.device("cpu"))
    ensemble = Ensemble([(_flip_sensitive_model(), metadata)])
    inputs = torch.zeros(1, 3, 8, 8)
    inputs[..., :4] = 1.0  # bright left half

    plain = ensemble.probabilities(inputs, flip=False)[0]
    flipped = ensemble.probabilities(inputs, flip=True)[0]

    assert plain[1] > 0.99
    assert flipped.tolist() == pytest.approx([0.5, 0.5])


def test_incompatible_members_are_refused(
    tmp_path: Path, write_checkpoint: CheckpointWriter
) -> None:
    with pytest.raises(ModelError, match="differ in class_names"):
        Ensemble.load(
            [write_checkpoint(GRADE_CLASSES, 0, True), write_checkpoint(QUALITY_CLASSES, 0, True)],
            torch.device("cpu"),
        )
    with pytest.raises(ModelError, match="at least one"):
        Ensemble([])
