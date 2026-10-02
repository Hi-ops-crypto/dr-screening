"""End-to-end: preprocess synthetic images, train a tiny model, evaluate it externally.

Uses the shipped configs with overrides that shrink everything (64 px images,
a test-sized backbone, two epochs on CPU), so it checks that the real configs,
cache, training loop, checkpoints and external evaluation fit together.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from drscreen.config import load_config
from drscreen.data.image_io import write_rgb
from drscreen.data.prepare import prepare_images
from drscreen.data.schema import GRADE, GROUP, IMAGE_ID, IMAGE_PATH, SOURCE, SPLIT, Source
from drscreen.data.settings import DataConfig
from drscreen.evaluation.external import EvaluationConfig, evaluate_checkpoint
from drscreen.evaluation.report import CONFUSION_FILE, METRICS_FILE
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
    """Write synthetic images whose brightness varies with grade."""
    rows = []
    for grade in range(5):
        for index in range(IMAGES_PER_GRADE):
            image_id = f"{source.value}_{grade}_{index}"
            path = root / f"{image_id}.png"
            colour = (60 + 35 * grade, 40, 20)
            write_rgb(make_fundus(height=80, width=100, radius=36, colour=colour), path)
            rows.append(
                {
                    IMAGE_ID: image_id,
                    IMAGE_PATH: str(path),
                    GRADE: grade,
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


def test_train_then_evaluate_externally(
    tmp_path: Path, make_fundus: FundusFactory, write_fundus: Callable[[Path], Path]
) -> None:
    data_config_path = tmp_path / "data.yaml"
    data = load_config(
        CONFIGS / "data.yaml",
        DataConfig,
        [
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.manifests_dir={tmp_path / 'manifests'}",
            "preprocessing.grading.image_size=64",
        ],
    )
    data_config_path.write_text(json.dumps(data.model_dump(mode="json")), encoding="utf-8")
    _prepare(
        data,
        Source.APTOS,
        _write_dataset(
            tmp_path / "raw_aptos", Source.APTOS, ["train", "train", "train", "val"], make_fundus
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
            f"data_config={data_config_path}",
            f"runs_dir={tmp_path / 'runs'}",
            "model.backbone=test_efficientnet",
            "model.pretrained=false",
            f"optimisation.epochs={EPOCHS}",
            "optimisation.batch_size=8",
            "optimisation.num_workers=0",
            "optimisation.device=cpu",
        ],
    )
    run_dir = run_training(training, data, REPO_ROOT)

    for name in (RUN_FILE, METRICS_LOG, BEST_CHECKPOINT, LAST_CHECKPOINT):
        assert (run_dir / name).is_file(), name
    epochs = [json.loads(line) for line in (run_dir / METRICS_LOG).read_text().splitlines()]
    assert [record["epoch"] for record in epochs] == list(range(1, EPOCHS + 1))
    assert all(np.isfinite(record["train_loss"]) for record in epochs)
    run_record = json.loads((run_dir / RUN_FILE).read_text())
    assert run_record["seed"] == training.optimisation.seed
    assert run_record["train_images"] == 15
    assert run_record["val_images"] == 5

    evaluation = load_config(
        CONFIGS / "evaluate.yaml",
        EvaluationConfig,
        ["num_workers=0", "device=cpu", "bootstrap.resamples=20"],
    )
    metrics = evaluate_checkpoint(run_dir / BEST_CHECKPOINT, evaluation, data)

    report_dir = run_dir / "eval-idrid"
    report = json.loads((report_dir / METRICS_FILE).read_text())
    assert report["metrics"] == metrics
    assert sum(map(sum, report["confusion_matrix"])) == 5 * IMAGES_PER_GRADE
    assert (report_dir / CONFUSION_FILE).is_file()
    assert "referable/sensitivity/ci_lower" in metrics
