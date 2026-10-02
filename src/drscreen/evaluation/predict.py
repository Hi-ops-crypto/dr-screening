"""Batch prediction over a data loader."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from drscreen.evaluation.metrics import Labels


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[Labels, Labels, float]:
    """Run a model over every batch and collect labels, predictions and loss.

    Args:
        model: Classifier; switched to evaluation mode.
        loader: Loader yielding ``(inputs, labels)`` batches.
        device: Device the model is on.

    Returns:
        ``(y_true, y_pred, mean_cross_entropy)``. The loss is unweighted so it
        is comparable across runs that weight classes differently.
    """
    model.eval()
    true_batches, predicted_batches = [], []
    loss_sum = 0.0
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        targets = labels.to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
        true_batches.append(labels.numpy())
        predicted_batches.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(true_batches).astype(np.int64)
    y_pred = np.concatenate(predicted_batches).astype(np.int64)
    return y_true, y_pred, loss_sum / len(y_true)
