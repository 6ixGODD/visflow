from __future__ import annotations

import typing as t

import numpy as np
import torch
import torchvision.datasets


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp augmentation to a batch of images and labels.

    Args:
        x (torch.Tensor): Batch of input images of shape (B, C, H, W).
        y (torch.Tensor): Batch of one-hot encoded labels of shape (B,
            num_classes).
        alpha (float): Parameter for the Beta distribution. Default is 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented images and mixed labels.
    """
    if alpha <= 0:
        return x, y

    batch_size = x.size(0)
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 for better mixing

    # Generate random permutation of indices
    index = torch.randperm(batch_size).to(x.device)

    # Create mixed inputs
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Create mixed labels
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y


def compute_class_weights(
    dataset: torchvision.datasets.DatasetFolder
) -> t.Dict[int, float]:
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_samples = len(dataset.samples)
    class_weights = {cls: total_samples / count
                     for cls, count in class_counts.items()}

    return class_weights
