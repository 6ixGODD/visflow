from __future__ import annotations

import typing as t

import numpy as np
import torch


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch of images and labels.

    Args:
        x (torch.Tensor): Batch of input images of shape (B, C, H, W).
        y (torch.Tensor): Batch of one-hot encoded labels of shape (B,
            num_classes).
        alpha (float): Parameter for the Beta distribution. Default is 1.0.

    Returns:
        mixed_x (torch.Tensor): Batch of mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Labels of the shuffled images.
        lam (float): MixUp coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def compute_class_weights(labels: t.Sequence[int]) -> t.Dict[int, float]:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor)
    total_samples = len(labels_tensor)

    class_weights = {
        cls: total_samples / count.item()  # type: ignore
        for cls, count in enumerate(class_counts)
        if count.item() > 0  # type: ignore
    }
    return class_weights
