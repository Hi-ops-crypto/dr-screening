"""Batch prediction over a data loader, for one model or an ensemble."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from drscreen.evaluation.metrics import Labels, Probabilities
from drscreen.models.ensemble import Ensemble


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[Labels, Probabilities, float]:
    """Run a model over every batch and collect labels, class probabilities and loss.

    Args:
        model: Classifier; switched to evaluation mode.
        loader: Loader yielding ``(inputs, labels)`` batches.
        device: Device the model is on.

    Returns:
        ``(y_true, probabilities, mean_cross_entropy)``, where probabilities
        has one softmax row per image; ``probabilities.argmax(axis=1)`` gives
        the predicted labels. The loss is unweighted so it is comparable
        across runs that weight classes differently.
    """
    model.eval()
    true_batches, probability_batches = [], []
    loss_sum = 0.0
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        targets = labels.to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
        true_batches.append(labels.numpy())
        probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(true_batches).astype(np.int64)
    probabilities = np.concatenate(probability_batches).astype(np.float64)
    return y_true, probabilities, loss_sum / len(y_true)


def predict_ensemble(
    ensemble: Ensemble,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    flip: bool,
) -> tuple[Labels, Probabilities]:
    """Collect labels and ensemble-averaged class probabilities for every batch.

    Args:
        ensemble: Loaded checkpoints sharing one input recipe.
        loader: Loader yielding ``(inputs, labels)`` batches, not shuffled.
        device: Device the ensemble is on.
        flip: Whether to also average over horizontally flipped inputs.

    Returns:
        ``(y_true, probabilities)`` with one softmax row per image.
    """
    true_batches, probability_batches = [], []
    for inputs, labels in loader:
        true_batches.append(labels.numpy())
        probability_batches.append(ensemble.probabilities(inputs.to(device), flip).cpu().numpy())
    y_true = np.concatenate(true_batches).astype(np.int64)
    return y_true, np.concatenate(probability_batches).astype(np.float64)
