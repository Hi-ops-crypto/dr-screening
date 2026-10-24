from pathlib import Path

import numpy as np
import pytest
import torch

from drscreen.data.preprocessing import crop_and_resize, normalise_lighting
from drscreen.data.tensors import image_to_tensor
from drscreen.exceptions import InputError, ModelError
from drscreen.inference.referral import referral_for
from drscreen.inference.screener import NO_RETINA_LABEL, GradingPolicy, Screener, ScreenerConfig
from drscreen.inference.validation import InputLimits, validate_upload
from tests.conftest import GRADE_CLASSES, QUALITY_CLASSES, CheckpointWriter, FundusFactory

LIMITS = InputLimits(min_side=64, max_pixels=1_000_000)
CONFIG = ScreenerConfig(
    device="cpu",
    limits=LIMITS,
    heatmap_alpha=0.4,
    grading=GradingPolicy(run_ids=["unused"], flip_tta=False, referral_threshold=0.3),
)


def _screener(
    quality: Path, grading: Path | list[Path], threshold: float = 0.3, flip: bool = False
) -> Screener:
    """Load a screener whose grading policy matches the grading checkpoints."""
    paths = grading if isinstance(grading, list) else [grading]
    policy = GradingPolicy(
        run_ids=[path.stem for path in paths], flip_tta=flip, referral_threshold=threshold
    )
    return Screener.from_checkpoints(quality, paths, CONFIG.model_copy(update={"grading": policy}))


def test_accepted_photo_is_graded_with_heatmap_and_referral(
    write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    screener = _screener(
        write_checkpoint(QUALITY_CLASSES, 1, False), write_checkpoint(GRADE_CLASSES, 3, True)
    )

    result = screener.screen(make_fundus())

    assert result.quality.label == "Usable"
    assert result.quality.accepted
    assert result.retake_message is None
    assert result.grade is not None
    assert result.grade.grade == 3
    assert result.grade.name == "Severe NPDR"
    assert result.grade.probabilities["Severe NPDR"] == pytest.approx(1.0)
    assert result.grade.heatmap.shape == (48, 48, 3)
    assert result.grade.referable_probability == pytest.approx(1.0)
    assert result.grade.referral == referral_for(3, 1.0, 0.3)


def test_rejected_photo_is_not_graded(
    write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    screener = _screener(
        write_checkpoint(QUALITY_CLASSES, 2, False), write_checkpoint(GRADE_CLASSES, 0, True)
    )

    result = screener.screen(make_fundus())

    assert result.quality.label == "Reject"
    assert not result.quality.accepted
    assert result.grade is None
    assert result.retake_message is not None
    assert "Retake" in result.retake_message


def test_photo_without_retina_asks_for_retake(write_checkpoint: CheckpointWriter) -> None:
    screener = _screener(
        write_checkpoint(QUALITY_CLASSES, 0, False), write_checkpoint(GRADE_CLASSES, 0, True)
    )

    result = screener.screen(np.zeros((200, 200, 3), dtype=np.uint8))

    assert result.quality.label == NO_RETINA_LABEL
    assert result.quality.confidence is None
    assert result.grade is None
    assert result.retake_message is not None
    assert "No retina" in result.retake_message


def test_checkpoints_with_wrong_classes_are_refused(write_checkpoint: CheckpointWriter) -> None:
    grading = write_checkpoint(GRADE_CLASSES, 0, True)

    with pytest.raises(ModelError, match="no 'Reject' class"):
        Screener.from_checkpoints(grading, [grading], CONFIG)
    with pytest.raises(ModelError, match="needs 5"):
        quality = write_checkpoint(QUALITY_CLASSES, 0, False)
        Screener.from_checkpoints(quality, [quality], CONFIG)


def test_threshold_tuned_for_another_run_is_refused(write_checkpoint: CheckpointWriter) -> None:
    with pytest.raises(ModelError, match="make tune-referral"):
        Screener.from_checkpoints(
            write_checkpoint(QUALITY_CLASSES, 0, False),
            [write_checkpoint(GRADE_CLASSES, 0, True)],
            CONFIG,
        )


def test_low_grade_is_referred_when_referable_probability_reaches_threshold(
    write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    screener = _screener(
        write_checkpoint(QUALITY_CLASSES, 0, False),
        write_checkpoint(GRADE_CLASSES, 0, True),
        threshold=0.0,
    )

    result = screener.screen(make_fundus())

    assert result.grade is not None
    assert result.grade.grade == 0
    assert result.grade.referral.refer
    assert "most likely grade is 0" in result.grade.referral.guidance


def test_grading_ensemble_averages_members_with_flips(
    write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    screener = _screener(
        write_checkpoint(QUALITY_CLASSES, 0, False),
        [write_checkpoint(GRADE_CLASSES, 1, True), write_checkpoint(GRADE_CLASSES, 4, True)],
        flip=True,
    )

    result = screener.screen(make_fundus())

    assert result.grade is not None
    assert result.grade.probabilities["Mild NPDR"] == pytest.approx(0.5)
    assert result.grade.probabilities["Proliferative DR"] == pytest.approx(0.5)
    assert result.grade.heatmap.shape == (48, 48, 3)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (None, "No image"),
        (np.zeros((100, 100), dtype=np.uint8), "not a colour photograph"),
        (np.zeros((100, 100, 3), dtype=np.float32), "not a colour photograph"),
        (np.zeros((40, 300, 3), dtype=np.uint8), "at least 64 pixels"),
        (np.zeros((1001, 1000, 3), dtype=np.uint8), "more than the 1,000,000"),
    ],
)
def test_unusable_uploads_are_rejected(image: np.ndarray | None, message: str) -> None:
    with pytest.raises(InputError, match=message):
        validate_upload(image, LIMITS)


@pytest.mark.parametrize(
    ("grade", "referable_probability", "refer", "guidance"),
    [
        (0, 0.1, False, "12 months"),
        (1, 0.29, False, "6 months"),
        (1, 0.3, True, "30% chance of moderate or worse"),
        (0, 0.6, True, "most likely grade is 0"),
        (2, 0.4, True, "ophthalmologist soon."),
        (3, 0.1, True, "ophthalmologist soon."),  # a referable grade is never un-referred
        (4, 0.9, True, "ophthalmologist soon."),
    ],
)
def test_referral_follows_grade_and_referable_probability(
    grade: int, referable_probability: float, refer: bool, guidance: str
) -> None:
    referral = referral_for(grade, referable_probability, threshold=0.3)

    assert referral.refer is refer
    assert guidance in referral.guidance


def test_referral_rejects_grades_outside_scale() -> None:
    with pytest.raises(ValueError, match="0 to 4"):
        referral_for(5, 0.0, 0.3)


def test_served_probabilities_match_the_evaluation_ensemble(
    tmp_path: Path, write_checkpoint: CheckpointWriter, make_fundus: FundusFactory
) -> None:
    graders = [write_checkpoint(GRADE_CLASSES, 0, True), write_checkpoint(GRADE_CLASSES, 1, True)]
    for path in graders:  # untrained heads, so outputs depend on the image and its flip
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["state_dict"] = {
            name: torch.randn_like(value) * 0.1 if name.startswith("classifier") else value
            for name, value in checkpoint["state_dict"].items()
        }
        torch.save(checkpoint, path)
    for flip in (False, True):
        screener = _screener(write_checkpoint(QUALITY_CLASSES, 0, False), graders, flip=flip)
        photo = make_fundus()
        recipe = screener._grading.preprocessing
        inputs = image_to_tensor(
            normalise_lighting(crop_and_resize(photo, recipe), recipe),
            screener._grading.mean,
            screener._grading.std,
        )[None]

        result = screener.screen(photo)

        assert result.grade is not None
        expected = screener._ensemble.probabilities(inputs, flip)[0].tolist()
        assert list(result.grade.probabilities.values()) == pytest.approx(expected, abs=1e-6)
