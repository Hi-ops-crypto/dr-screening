"""Screen one fundus photograph: quality gate, then grade, heatmap and referral.

Each model is fed with the preprocessing recipe and input statistics stored in
its own checkpoint, exactly as it was trained. The quality model sees the raw
crop. Grading uses one or more checkpoints whose probabilities are averaged
(``drscreen.models.ensemble``), optionally with flip test-time augmentation;
they see the lighting-normalised crop, and their Grad-CAM heatmaps are
averaged and drawn over the un-normalised crop so it reads like a photograph.
"""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydantic import Field

from drscreen.config import StrictModel
from drscreen.data.preprocessing import (
    Image,
    crop_and_resize,
    fundus_mask,
    normalise_lighting,
    preprocess,
)
from drscreen.data.tensors import image_to_tensor
from drscreen.devices import DeviceName, resolve_device
from drscreen.exceptions import FundusNotFoundError, ModelError
from drscreen.explain.gradcam import (
    forward_with_activations,
    heatmap_for,
    last_conv_layer,
    overlay,
)
from drscreen.inference.referral import MAX_GRADE, REFERABLE_MIN_GRADE, Referral, referral_for
from drscreen.inference.validation import InputLimits, validate_upload
from drscreen.models.checkpoint import CheckpointMetadata, load_checkpoint
from drscreen.models.ensemble import Ensemble

logger = logging.getLogger(__name__)

REJECT_CLASS = "Reject"  # EyeQ's label for ungradable images
NO_RETINA_LABEL = "No retina found"


class GradingPolicy(StrictModel):
    """The grading checkpoints that serve together and the threshold tuned for them.

    The referral threshold from ``make tune-referral`` is only valid for the
    exact runs and flip setting it was tuned with, so all three live together.
    """

    run_ids: list[str] = Field(min_length=1)
    flip_tta: bool
    referral_threshold: float = Field(ge=0, le=1)


class ScreenerConfig(StrictModel):
    """Serving settings that are not stored in the checkpoints."""

    device: DeviceName
    limits: InputLimits
    heatmap_alpha: float = Field(ge=0, le=1)
    grading: GradingPolicy


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the quality gate.

    Attributes:
        label: Predicted quality class, or ``NO_RETINA_LABEL``.
        confidence: Softmax probability of ``label``; ``None`` when no
            retina was found and the model never ran.
        accepted: Whether the image goes on to grading.
    """

    label: str
    confidence: float | None
    accepted: bool


@dataclass(frozen=True)
class GradeResult:
    """Outcome of grading an accepted image.

    Attributes:
        grade: ICDR grade from 0 to 4.
        name: Display name of the grade.
        probabilities: Averaged softmax probability of every grade, by name.
            Not calibrated: read it as the model's ranking, not a likelihood.
        referable_probability: Summed probability of grade 2 or worse.
        photo: The cropped photograph the grade was made from, before
            lighting normalisation.
        heatmap: Averaged Grad-CAM overlay on ``photo``.
        referral: Screening guidance for the grade.
    """

    grade: int
    name: str
    probabilities: dict[str, float]
    referable_probability: float
    photo: Image
    heatmap: Image
    referral: Referral


@dataclass(frozen=True)
class ScreeningResult:
    """Everything the app shows for one photograph.

    Attributes:
        quality: Quality gate outcome.
        grade: Grading outcome, or ``None`` if the image was rejected.
        retake_message: Why the image was rejected, or ``None`` if it was graded.
    """

    quality: QualityResult
    grade: GradeResult | None
    retake_message: str | None


class Screener:
    """Runs the quality model and the grading ensemble on single photographs.

    Calls to ``screen`` are serialised: Grad-CAM attaches hooks to the shared
    grading models, and one CPU gains nothing from running two images at once.
    """

    def __init__(
        self,
        quality: tuple[torch.nn.Module, CheckpointMetadata],
        grading: Ensemble,
        config: ScreenerConfig,
        device: torch.device,
    ) -> None:
        """Wrap the loaded models.

        Args:
            quality: Quality model and its metadata.
            grading: Grading checkpoints as an ensemble.
            config: Serving settings.
            device: Device all models are on.

        Raises:
            ModelError: If the checkpoints do not predict the expected classes,
                or the referral threshold was tuned for other grading runs.
        """
        self._quality_model, self._quality = quality
        self._ensemble = grading
        self._grading = grading.metadata
        if REJECT_CLASS not in self._quality.class_names:
            raise ModelError(
                f"Quality checkpoint {self._quality.run_id} predicts "
                f"{self._quality.class_names}, which has no {REJECT_CLASS!r} class."
            )
        if len(self._grading.class_names) != MAX_GRADE + 1:
            raise ModelError(
                f"Grading checkpoint {self._grading.run_id} predicts "
                f"{len(self._grading.class_names)} classes; ICDR grading needs {MAX_GRADE + 1}."
            )
        if config.grading.run_ids != grading.run_ids:
            raise ModelError(
                f"The referral threshold in the config was tuned for grading runs "
                f"{config.grading.run_ids}, but the checkpoints are {grading.run_ids}. "
                "Run `make tune-referral` for these checkpoints."
            )
        self._config = config
        self._device = device
        self._layers = [last_conv_layer(model) for model in grading.models]
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoints(
        cls, quality: Path, grading: Sequence[Path], config: ScreenerConfig
    ) -> "Screener":
        """Load all models from checkpoint files.

        Args:
            quality: Quality model checkpoint.
            grading: One or more grading checkpoints, averaged as an ensemble.
            config: Serving settings.

        Returns:
            A ready screener.

        Raises:
            ModelError: If a checkpoint is missing, unreadable or predicts the wrong
                classes, or the referral threshold belongs to other grading runs.
        """
        device = resolve_device(config.device)
        screener = cls(
            load_checkpoint(quality, device), Ensemble.load(grading, device), config, device
        )
        logger.info(
            "Loaded quality model %s and grading models %s (flip_tta=%s)",
            screener._quality.run_id,
            "+".join(screener._ensemble.run_ids),
            config.grading.flip_tta,
        )
        return screener

    def screen(self, image: npt.NDArray[np.generic] | None) -> ScreeningResult:
        """Screen one photograph.

        Args:
            image: Decoded RGB upload.

        Returns:
            The quality outcome and, if accepted, the grade.

        Raises:
            InputError: If the upload is missing, not RGB or of an unusable size.
        """
        photo = validate_upload(image, self._config.limits)
        with self._lock:
            try:
                quality = self._assess_quality(photo)
                if not quality.accepted:
                    return ScreeningResult(quality, None, _quality_retake_message(quality))
                return ScreeningResult(quality, self._grade(photo), None)
            except FundusNotFoundError as exc:
                logger.info("No retina found: %s", exc)
                return ScreeningResult(
                    QualityResult(NO_RETINA_LABEL, None, accepted=False),
                    None,
                    "No retina was found in the photo. Make sure the whole retina is in the "
                    "frame and well lit, then retake it.",
                )

    def _assess_quality(self, photo: Image) -> QualityResult:
        """Run the quality model on the raw crop.

        Args:
            photo: Validated RGB photograph.

        Returns:
            The predicted quality class and whether it passes.

        Raises:
            FundusNotFoundError: If no retina can be located.
        """
        recipe = self._quality.preprocessing
        inputs = self._to_batch(preprocess(photo, recipe), self._quality)
        with torch.no_grad():
            probabilities = torch.softmax(self._quality_model(inputs)[0], dim=0)
        index = int(probabilities.argmax())
        label = self._quality.class_names[index]
        return QualityResult(label, float(probabilities[index]), accepted=label != REJECT_CLASS)

    def _grade(self, photo: Image) -> GradeResult:
        """Grade the normalised crop and explain the prediction.

        Args:
            photo: Validated RGB photograph that passed the quality gate.

        Returns:
            Grade, probabilities, heatmap overlay and referral.

        Raises:
            FundusNotFoundError: If no retina can be located.
        """
        recipe = self._grading.preprocessing
        policy = self._config.grading
        crop = crop_and_resize(photo, recipe)
        inputs = self._to_batch(normalise_lighting(crop, recipe), self._grading)
        # Each member sees the image once with gradients, which gives both its
        # prediction and the activations its heatmap needs, and the mirror image
        # (if used) once without, so nothing is computed twice.
        passes = [
            forward_with_activations(model, inputs, layer)
            for model, layer in zip(self._ensemble.models, self._layers, strict=True)
        ]
        member_logits = [logits for logits, _ in passes]
        if policy.flip_tta:
            mirrored = self._ensemble.views(inputs, flip=True)[len(inputs) :]
            with torch.no_grad():
                member_logits = [
                    torch.cat([logits.detach(), model(mirrored)])
                    for logits, model in zip(member_logits, self._ensemble.models, strict=True)
                ]
        probabilities = self._ensemble.combine(member_logits, len(inputs))[0].tolist()
        referable_probability = sum(probabilities[REFERABLE_MIN_GRADE:])
        grade = max(range(len(probabilities)), key=probabilities.__getitem__)
        size = (inputs.shape[-2], inputs.shape[-1])
        heatmap = np.mean(
            [heatmap_for(logits[0, grade], activations, size) for logits, activations in passes],
            axis=0,
        ).astype(np.float32)
        heatmap *= fundus_mask(
            crop, recipe.fundus_intensity_threshold
        )  # nothing outside the retina
        return GradeResult(
            grade=grade,
            name=self._grading.class_names[grade],
            probabilities=dict(zip(self._grading.class_names, probabilities, strict=True)),
            referable_probability=referable_probability,
            photo=crop,
            heatmap=overlay(crop, heatmap, self._config.heatmap_alpha),
            referral=referral_for(grade, referable_probability, policy.referral_threshold),
        )

    def _to_batch(self, image: Image, metadata: CheckpointMetadata) -> torch.Tensor:
        """Turn a preprocessed image into a one-image batch on the model's device.

        Args:
            image: Preprocessed RGB image.
            metadata: Checkpoint whose input statistics to apply.

        Returns:
            A float tensor of shape ``(1, 3, H, W)``.
        """
        return image_to_tensor(image, metadata.mean, metadata.std)[None].to(self._device)


def _quality_retake_message(quality: QualityResult) -> str:
    """Explain a quality rejection and what to change.

    Args:
        quality: A rejected quality outcome from the model.

    Returns:
        The message.
    """
    return (
        f"The photograph is not clear enough to grade (quality check: {quality.label}, "
        f"{quality.confidence:.0%} confidence). Common causes are blur, too little or too "
        "much light, eyelashes or reflections across the retina, or the retina not fully "
        "in frame. Retake it."
    )
