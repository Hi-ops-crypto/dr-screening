"""End-to-end: preprocess synthetic images, train tiny models, evaluate externally.

Uses the shipped configs with overrides that shrink everything (64 px images,
a test-sized backbone, two epochs on CPU), so it checks that the real configs,
cache, training loop, checkpoints and external evaluation fit together.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drscreen.config import load_config
from drscreen.data.image_io import write_rgb
from drscreen.data.prepare import prepare_images
from drscreen.data.schema import (
    GRADE,
    GROUP,
    IMAGE_ID,
    IMAGE_PATH,
    LABEL_COLUMN,
    SOURCE,
    SPLIT,
    Source,
)
from drscreen.data.settings import DataConfig
from drscreen.evaluation.external import (
    THRESHOLD_FILE,
    EvaluationConfig,
    evaluate_checkpoint,
    tune_referral_threshold,
)
from drscreen.evaluation.report import CONFUSION_FILE, METRICS_FILE
from drscreen.exceptions import DataError, ModelError
from drscreen.training.settings import TrainingRunConfig
from drscreen.training.trainer import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    METRICS_LOG,
    RUN_FILE,
    run_training,
)
from tests.conftest import FundusFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
IMAGES_PER_GRADE = 4
EPOCHS = 2


def _write_dataset(
    root: Path, source: Source, splits: list[str], make_fundus: FundusFactory
) -> pd.DataFrame:
    """Write synthetic images whose brightness varies with the label."""
    label_column = LABEL_COLUMN[source]
    rows = []
    for grade in range(5 if label_column == GRADE else 3):
        for index in range(IMAGES_PER_GRADE):
            image_id = f"{source.value}_{grade}_{index}"
            path = root / f"{image_id}.png"
            colour = (60 + 35 * grade, 40, 20)
            write_rgb(make_fundus(height=80, width=100, radius=36, colour=colour), path)
            rows.append(
                {
                    IMAGE_ID: image_id,
                    IMAGE_PATH: str(path),
                    label_column: grade,
                    SOURCE: source.value,
                    GROUP: image_id,
                    SPLIT: splits[index % len(splits)],
                }
            )
    return pd.DataFrame(rows)


def _prepare(data: DataConfig, source: Source, raw: pd.DataFrame) -> None:
    output = data.paths.processed_manifest(source)
    result = prepare_images(raw, output.parent, data.preprocessing.for_source(source), workers=1)
    assert result.failures.empty
    result.manifest.to_csv(output, index=False)


def _shrunk_data_config(tmp_path: Path) -> tuple[DataConfig, Path]:
    """Load the shipped data config pointed at ``tmp_path`` with 64 px images."""
    data = load_config(
        CONFIGS / "data.yaml",
        DataConfig,
        [
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.manifests_dir={tmp_path / 'manifests'}",
            "preprocessing.grading.image_size=64",
            "preprocessing.quality.image_size=64",
        ],
    )
    path = tmp_path / "data.yaml"
    path.write_text(json.dumps(data.model_dump(mode="json")), encoding="utf-8")
    return data, path


def _shrunk_training_overrides(data_config_path: Path, runs_dir: Path) -> list[str]:
    return [
        f"data_config={data_config_path}",
        f"runs_dir={runs_dir}",
        "model.backbone=test_efficientnet",
        "model.pretrained=false",
        f"optimisation.epochs={EPOCHS}",
        "optimisation.batch_size=8",
        "optimisation.num_workers=0",
        "optimisation.device=cpu",
    ]


def test_train_quality_model(tmp_path: Path, make_fundus: FundusFactory) -> None:
    data, data_config_path = _shrunk_data_config(tmp_path)
    _prepare(
        data,
        Source.EYEQ,
        _write_dataset(tmp_path / "raw_eyeq", Source.EYEQ, ["train", "train", "val"], make_fundus),
    )
    training = load_config(
        CONFIGS / "train_quality.yaml",
        TrainingRunConfig,
        _shrunk_training_overrides(data_config_path, tmp_path / "runs"),
    )

    run_dir = run_training(training, data, REPO_ROOT)

    epochs = [json.loads(line) for line in (run_dir / METRICS_LOG).read_text().splitlines()]
    assert run_dir.name.startswith("eyeq-")
    assert (run_dir / BEST_CHECKPOINT).is_file()
    assert "sensitivity/reject" in epochs[-1]
    assert not any(key.startswith("referable/") for key in epochs[-1])
    run_record = json.loads((run_dir / RUN_FILE).read_text())
    assert run_record["preprocessing"]["ben_graham"] is None
    assert len(run_record["class_weights"]) == 3


def test_train_then_evaluate_externally(
    tmp_path: Path, make_fundus: FundusFactory, write_fundus: Callable[[Path], Path]
) -> None:
    data, data_config_path = _shrunk_data_config(tmp_path)
    for source in (Source.APTOS, Source.EYEPACS):
        _prepare(
            data,
            source,
            _write_dataset(
                tmp_path / f"raw_{source.value}",
                source,
                ["train", "train", "train", "val"],
                make_fundus,
            ),
        )
    _prepare(
        data,
        Source.IDRID,
        _write_dataset(tmp_path / "raw_idrid", Source.IDRID, ["external"], make_fundus),
    )

    training = load_config(
        CONFIGS / "train_grading.yaml",
        TrainingRunConfig,
        [
            *_shrunk_training_overrides(data_config_path, tmp_path / "runs"),
            "sources=[aptos, eyepacs]",
        ],
    )
    with pytest.raises(DataError, match="no training images"):
        run_training(
            training.model_copy(update={"sources": [Source.APTOS, Source.IDRID]}), data, REPO_ROOT
        )
    run_dir = run_training(training, data, REPO_ROOT)

    for name in (RUN_FILE, METRICS_LOG, BEST_CHECKPOINT, LAST_CHECKPOINT):
        assert (run_dir / name).is_file(), name
    epochs = [json.loads(line) for line in (run_dir / METRICS_LOG).read_text().splitlines()]
    assert [record["epoch"] for record in epochs] == list(range(1, EPOCHS + 1))
    assert all(np.isfinite(record["train_loss"]) for record in epochs)
    run_record = json.loads((run_dir / RUN_FILE).read_text())
    assert run_record["seed"] == training.optimisation.seed
    assert run_dir.name.startswith("aptos+eyepacs-")
    assert run_record["train_images"] == 30
    assert run_record["val_images"] == 10

    evaluation = load_config(
        CONFIGS / "evaluate.yaml",
        EvaluationConfig,
        [
            "num_workers=0",
            "device=cpu",
            "bootstrap.resamples=20",
            "referral.tuning_sources=[aptos, eyepacs]",
        ],
    )
    metrics = evaluate_checkpoint([run_dir / BEST_CHECKPOINT], evaluation, data, run_dir)

    report_dir = run_dir / "eval-idrid"
    report = json.loads((report_dir / METRICS_FILE).read_text())
    assert report["metrics"] == metrics
    assert sum(map(sum, report["confusion_matrix"])) == 5 * IMAGES_PER_GRADE
    assert (report_dir / CONFUSION_FILE).is_file()
    assert "referable/sensitivity/ci_lower" in metrics
    assert not any(key.startswith("referable@threshold/") for key in metrics)

    tuned = tune_referral_threshold([run_dir / BEST_CHECKPOINT], evaluation, data, run_dir)
    assert tuned["grading_run_ids"] == [run_dir.name]
    assert tuned["tuning_images"] == 10
    assert (run_dir / THRESHOLD_FILE).is_file()

    with_threshold = load_config(
        CONFIGS / "evaluate.yaml",
        EvaluationConfig,
        [
            "num_workers=0",
            "device=cpu",
            "bootstrap.resamples=20",
            "referral.tuning_sources=[aptos, eyepacs]",
            f"referral.threshold={tuned['threshold']}",
            "flip_tta=true",
        ],
    )
    ensemble_dir = tmp_path / "ensemble"
    thresholded = evaluate_checkpoint(
        [run_dir / BEST_CHECKPOINT, run_dir / LAST_CHECKPOINT], with_threshold, data, ensemble_dir
    )
    assert (ensemble_dir / "eval-idrid" / METRICS_FILE).is_file()
    assert thresholded["referable@threshold/threshold"] == tuned["threshold"]
    assert "referable@threshold/sensitivity/ci_upper" in thresholded


def test_site_adaptation_fine_tunes_on_one_split_and_scores_another(
    tmp_path: Path, make_fundus: FundusFactory
) -> None:
    data, data_config_path = _shrunk_data_config(tmp_path)
    # Site adaptation: official training images split into train/val, test images external.
    _prepare(
        data,
        Source.IDRID,
        _write_dataset(
            tmp_path / "raw_idrid", Source.IDRID, ["train", "train", "val", "external"], make_fundus
        ),
    )
    _prepare(
        data,
        Source.APTOS,
        _write_dataset(tmp_path / "raw_aptos", Source.APTOS, ["train", "val"], make_fundus),
    )
    overrides = _shrunk_training_overrides(data_config_path, tmp_path / "runs")
    base = run_training(
        load_config(
            CONFIGS / "train_grading.yaml", TrainingRunConfig, [*overrides, "sources=[aptos]"]
        ),
        data,
        REPO_ROOT,
    )

    adapted = run_training(
        load_config(
            CONFIGS / "finetune_site.yaml",
            TrainingRunConfig,
            [
                *overrides,
                f"init_checkpoint={base / BEST_CHECKPOINT}",
            ],
        ),
        data,
        REPO_ROOT,
    )

    record = json.loads((adapted / RUN_FILE).read_text())
    assert record["init_run_id"] == base.name
    assert record["train_images"] == 10
    evaluation = load_config(
        CONFIGS / "evaluate.yaml",
        EvaluationConfig,
        [
            "num_workers=0",
            "device=cpu",
            "bootstrap.resamples=20",
            "referral.tuning_sources=[idrid]",
        ],
    )
    tuned = tune_referral_threshold([adapted / BEST_CHECKPOINT], evaluation, data, adapted)
    assert tuned["tuning_images"] == 5
    metrics = evaluate_checkpoint([adapted / BEST_CHECKPOINT], evaluation, data, adapted)
    report = json.loads((adapted / "eval-idrid" / METRICS_FILE).read_text())
    assert report["metrics"] == metrics
    assert sum(map(sum, report["confusion_matrix"])) == 5


def test_fine_tuning_refuses_a_different_backbone(
    tmp_path: Path, make_fundus: FundusFactory
) -> None:
    data, data_config_path = _shrunk_data_config(tmp_path)
    _prepare(
        data,
        Source.APTOS,
        _write_dataset(tmp_path / "raw_aptos", Source.APTOS, ["train", "val"], make_fundus),
    )
    overrides = _shrunk_training_overrides(data_config_path, tmp_path / "runs")
    base = run_training(
        load_config(
            CONFIGS / "train_grading.yaml", TrainingRunConfig, [*overrides, "sources=[aptos]"]
        ),
        data,
        REPO_ROOT,
    )
    other = load_config(
        CONFIGS / "train_grading.yaml",
        TrainingRunConfig,
        [
            *overrides,
            "sources=[aptos]",
            "model.backbone=efficientnet_b0",
            f"init_checkpoint={base / BEST_CHECKPOINT}",
        ],
    )

    with pytest.raises(ModelError, match="backbone"):
        run_training(other, data, REPO_ROOT)
