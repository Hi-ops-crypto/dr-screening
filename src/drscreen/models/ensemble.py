"""Average the predictions of several checkpoints, optionally with flip test-time augmentation.

Members must share class names, preprocessing recipe and input statistics, so
one input tensor serves them all. Averaging softmax probabilities of models
that make different mistakes usually beats each of them; averaging each image
with its mirror image (left and right eyes are mirror images, so a flipped
fundus is a realistic input) removes some orientation-specific noise.

Evaluation and the web app both get probabilities from ``Ensemble``, so the
two cannot combine members differently.
"""

from collections.abc import Sequence
from pathlib import Path

import torch

from drscreen.exceptions import ModelError
from drscreen.models.checkpoint import CheckpointMetadata, load_checkpoint

WIDTH_DIM = -1


class Ensemble:
    """One or more loaded checkpoints that share one input recipe."""

    def __init__(self, members: Sequence[tuple[torch.nn.Module, CheckpointMetadata]]) -> None:
        """Wrap loaded models.

        Args:
            members: Models in evaluation mode with their metadata.

        Raises:
            ModelError: If there are no members or they disagree on classes,
                preprocessing or input statistics.
        """
        if not members:
            raise ModelError("An ensemble needs at least one checkpoint.")
        first = members[0][1]
        for _, metadata in members[1:]:
            for field in ("class_names", "preprocessing", "mean", "std"):
                if getattr(metadata, field) != getattr(first, field):
                    raise ModelError(
                        f"Checkpoints {first.run_id} and {metadata.run_id} differ in {field}; "
                        "ensemble members must share classes and inputs."
                    )
        self.models = [model for model, _ in members]
        self.metadata = first
        self.run_ids = [metadata.run_id for _, metadata in members]

    @classmethod
    def load(cls, paths: Sequence[Path], device: torch.device) -> "Ensemble":
        """Load every checkpoint onto a device.

        Args:
            paths: Checkpoint files.
            device: Device to load onto.

        Returns:
            The ensemble.

        Raises:
            ModelError: If a checkpoint is unusable or the members are incompatible.
        """
        return cls([load_checkpoint(path, device) for path in paths])

    @staticmethod
    def views(inputs: torch.Tensor, flip: bool) -> torch.Tensor:
        """Return the batch every member is run on: the inputs, then their mirror images.

        Args:
            inputs: Float tensor of shape ``(N, 3, H, W)``.
            flip: Whether to append horizontally flipped copies.

        Returns:
            ``inputs``, or a ``(2N, 3, H, W)`` batch with the flipped copies last.
        """
        return torch.cat([inputs, inputs.flip(WIDTH_DIM)]) if flip else inputs

    def combine(self, member_logits: Sequence[torch.Tensor], num_images: int) -> torch.Tensor:
        """Average softmax probabilities over members and views.

        Args:
            member_logits: One logits tensor per member, each computed on
                ``views(inputs, flip)`` for the same ``num_images`` inputs.
            num_images: Number of original images.

        Returns:
            Tensor of shape ``(num_images, num_classes)`` whose rows sum to 1.
        """
        per_member = [
            torch.softmax(logits.detach(), dim=1).view(-1, num_images, logits.shape[1]).mean(0)
            for logits in member_logits
        ]
        return torch.stack(per_member).mean(0)

    @torch.no_grad()
    def probabilities(self, inputs: torch.Tensor, flip: bool) -> torch.Tensor:
        """Average softmax probabilities over members and, optionally, flips.

        Args:
            inputs: Float tensor of shape ``(N, 3, H, W)`` on the members' device.
            flip: Whether to also average over horizontally flipped inputs.

        Returns:
            Tensor of shape ``(N, num_classes)`` whose rows sum to 1.
        """
        batch = self.views(inputs, flip)
        return self.combine([model(batch) for model in self.models], len(inputs))
