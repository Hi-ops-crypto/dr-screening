from pathlib import Path

import numpy as np
import pytest
import torch

from drscreen.config import load_config
from drscreen.data.schema import Source
from drscreen.evaluation.external import EvaluationConfig
from drscreen.exceptions import ConfigError, DataError
from drscreen.reproducibility import git_state, seed_everything
from drscreen.training.settings import TrainingRunConfig
from drscreen.training.trainer import class_weights

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_class_weights_are_inverse_frequency() -> None:
    weights = class_weights(np.array([0, 0, 0, 1]), num_classes=2, power=1.0)

    assert weights.tolist() == pytest.approx([4 / 6, 4 / 2])


def test_class_weight_power_softens_or_removes_weighting() -> None:
    labels = np.array([0, 0, 0, 1])

    assert class_weights(labels, 2, power=0.5).tolist() == pytest.approx([(4 / 6) ** 0.5, 2**0.5])
    assert class_weights(labels, 2, power=0.0).tolist() == pytest.approx([1.0, 1.0])


def test_class_weights_reject_missing_class() -> None:
    with pytest.raises(DataError, match=r"Classes \[2\]"):
        class_weights(np.array([0, 1]), num_classes=3, power=1.0)


def test_shipped_training_and_evaluation_configs_are_valid() -> None:
    training = load_config(REPO_ROOT / "configs" / "train_grading.yaml", TrainingRunConfig)
    evaluation = load_config(REPO_ROOT / "configs" / "evaluate.yaml", EvaluationConfig)

    assert training.sources == evaluation.referral.tuning_sources
    assert Source.IDRID not in training.sources
    assert evaluation.source.value == "idrid"
    assert training.referable_min_grade == evaluation.referable_min_grade


def test_shipped_quality_config_trains_eyeq_without_referral() -> None:
    quality = load_config(REPO_ROOT / "configs" / "train_quality.yaml", TrainingRunConfig)

    assert quality.sources == [Source.EYEQ]
    assert quality.referable_min_grade is None
    assert quality.augmentation.blur_probability == 0
    assert quality.augmentation.vignette_probability == 0


def test_training_on_differently_labelled_sources_is_refused() -> None:
    with pytest.raises(ConfigError, match="labelled differently"):
        load_config(
            REPO_ROOT / "configs" / "train_grading.yaml",
            TrainingRunConfig,
            ["sources=[aptos, eyeq]"],
        )


def test_seed_everything_makes_torch_repeatable() -> None:
    seed_everything(3, deterministic=True)
    first = torch.rand(3)
    seed_everything(3, deterministic=True)

    assert torch.equal(first, torch.rand(3))


def test_git_state_outside_repository_is_unknown(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_git_state_outside_repository_reads_commit_file(tmp_path: Path) -> None:
    (tmp_path / "GIT_COMMIT").write_text("abc123\n")

    assert git_state(tmp_path) == {"commit": "abc123", "dirty": None}


def test_git_state_inside_repository_has_commit() -> None:
    state = git_state(REPO_ROOT)

    assert isinstance(state["commit"], str)
    assert len(state["commit"]) == 40
