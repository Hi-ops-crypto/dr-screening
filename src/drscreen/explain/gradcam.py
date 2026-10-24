"""Grad-CAM heatmaps for convolutional classifiers.

R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017. The activations of a convolutional
layer are weighted by the spatially averaged gradient of the target class
score, summed over channels and rectified, giving a coarse map of the regions
that pushed the score up. It shows where the model looked, not where lesions
are.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
import torch

Heatmap = npt.NDArray[np.float32]

UINT8_MAX = 255


@dataclass(frozen=True)
class GradCamResult:
    """A heatmap together with the logits of the forward pass that produced it.

    Attributes:
        heatmap: Float32 array of shape ``(H, W)`` matching the input, scaled
            to ``[0, 1]``; all zeros when no region raised the class score.
        logits: Raw class scores of shape ``(num_classes,)``.
        target_class: Class the heatmap explains.
    """

    heatmap: Heatmap
    logits: torch.Tensor
    target_class: int


def last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last ``Conv2d`` module registered in a model.

    For timm CNNs this is the final convolution before pooling (for example
    EfficientNet's ``conv_head``), the usual Grad-CAM target.

    Args:
        model: A convolutional classifier.

    Returns:
        The layer.

    Raises:
        ValueError: If the model has no ``Conv2d`` layer.
    """
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convolutions:
        raise ValueError(f"{type(model).__name__} has no Conv2d layer to explain.")
    return convolutions[-1]


def forward_with_activations(
    model: torch.nn.Module, inputs: torch.Tensor, layer: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass with gradients enabled, capturing a layer's output.

    Callers that need predictions and heatmaps from the same images use this
    once and then call ``heatmap_for``, instead of running the model twice.

    Args:
        model: Classifier in evaluation mode.
        inputs: Float tensor of shape ``(N, 3, H, W)`` on the model's device.
        layer: Convolutional layer inside ``model`` to explain.

    Returns:
        ``(logits, activations)``, both still attached to the autograd graph.
    """
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _module, _args, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = model(inputs.detach().requires_grad_(True))
    finally:
        handle.remove()
    return logits, captured[0]


def heatmap_for(
    score: torch.Tensor, activations: torch.Tensor, size: tuple[int, int], index: int = 0
) -> Heatmap:
    """Compute the Grad-CAM heatmap of one image from a class score.

    Gradients are taken with respect to the activations only, so no parameter
    ``.grad`` is written and the model is left unchanged.

    Args:
        score: Scalar logit to explain, from ``forward_with_activations``.
        activations: Activations captured in the same forward pass.
        size: ``(H, W)`` of the input image.
        index: Position of the image in the batch.

    Returns:
        Float32 heatmap of shape ``size`` scaled to ``[0, 1]``.
    """
    (gradients,) = torch.autograd.grad(score, activations)
    weights = gradients[index].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations.detach()[index]).sum(dim=0))
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0]
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    heatmap: Heatmap = cam.cpu().numpy().astype(np.float32)
    return heatmap


def gradcam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    layer: torch.nn.Module,
    target_class: int | None = None,
) -> GradCamResult:
    """Run one forward pass and compute the Grad-CAM heatmap for one image.

    Args:
        model: Classifier in evaluation mode.
        inputs: One image as a float tensor of shape ``(1, 3, H, W)`` on the
            model's device.
        layer: Convolutional layer inside ``model`` to explain.
        target_class: Class to explain; the predicted class when ``None``.

    Returns:
        The heatmap, logits and explained class.
    """
    logits, activations = forward_with_activations(model, inputs, layer)
    target = int(logits[0].argmax()) if target_class is None else target_class
    heatmap = heatmap_for(logits[0, target], activations, (inputs.shape[-2], inputs.shape[-1]))
    return GradCamResult(heatmap=heatmap, logits=logits[0].detach(), target_class=target)


def overlay(image: npt.NDArray[np.uint8], heatmap: Heatmap, alpha: float) -> npt.NDArray[np.uint8]:
    """Blend a heatmap, coloured with the jet colour map, over an RGB image.

    Each pixel's opacity is ``alpha * heatmap``, so regions the model ignored
    show the photograph unchanged and only the regions that drove the
    prediction are coloured.

    Args:
        image: RGB uint8 array of shape ``(H, W, 3)``.
        heatmap: Values in ``[0, 1]``; resized to the image if shapes differ.
        alpha: Opacity where the heatmap is 1, in ``[0, 1]``.

    Returns:
        The blended RGB uint8 image.
    """
    height, width = image.shape[:2]
    if heatmap.shape != (height, width):
        resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = np.asarray(resized, dtype=np.float32)
    levels = np.clip(np.rint(heatmap * UINT8_MAX), 0, UINT8_MAX).astype(np.uint8)
    colours = cv2.cvtColor(cv2.applyColorMap(levels, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    weight = (alpha * np.clip(heatmap, 0, 1))[..., None]
    blended = colours * weight + image * (1 - weight)
    return np.clip(np.rint(blended), 0, UINT8_MAX).astype(np.uint8)
