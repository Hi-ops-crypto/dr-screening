from pathlib import Path

import pytest
import torch

from drscreen.data.preprocessing import PreprocessingConfig
from drscreen.exceptions import ModelError
from drscreen.models.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from drscreen.models.factory import ModelConfig, build_classifier, input_stats

TINY = ModelConfig(backbone="test_efficientnet", pretrained=False, weights_path=None, drop_rate=0.0)


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        run_id="test",
        epoch=1,
        model=TINY,
        class_names=("a", "b", "c"),
        preprocessing=PreprocessingConfig(
            image_size=32, fundus_intensity_threshold=7, min_fundus_fraction=0.1, ben_graham=None
        ),
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
        metrics={"qwk": 0.5, "sensitivity/a": None},
    )


def test_builds_classifier_with_requested_classes() -> None:
    model = build_classifier(TINY, num_classes=5)

    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 5)


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ModelError, match="Unknown backbone"):
        build_classifier(TINY.model_copy(update={"backbone": "no_such_net"}), 5)


def test_missing_weights_file_is_rejected(tmp_path: Path) -> None:
    config = TINY.model_copy(update={"weights_path": tmp_path / "absent.pth"})

    with pytest.raises(ModelError, match="not found"):
        build_classifier(config, 5)


def test_input_stats_come_from_backbone() -> None:
    mean, std = input_stats(build_classifier(TINY, 5))

    assert len(mean) == len(std) == 3
    assert all(value > 0 for value in std)


def test_checkpoint_round_trip_restores_weights_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_classifier(TINY, 3).eval()
    inputs = torch.rand(1, 3, 32, 32)
    path = tmp_path / "best.pt"

    save_checkpoint(path, model, _metadata())
    restored, metadata = load_checkpoint(path, torch.device("cpu"))

    assert metadata == _metadata()
    assert not restored.training
    assert torch.allclose(model(inputs), restored(inputs))
    assert not path.with_name("best.pt.partial").exists()


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt", torch.device("cpu"))


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(ModelError, match="corrupt"):
        load_checkpoint(path, torch.device("cpu"))
