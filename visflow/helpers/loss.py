from __future__ import annotations

import typing as t

import torch
import torch.nn as nn


class MixUpLoss(nn.Module):
    def __init__(
        self,
        criterion: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        pred: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor | None = None,
        lam: float | None = None
    ) -> torch.Tensor:
        if target_b is not None and lam is not None:
            return (lam *
                    self.criterion(pred, target_a) + (1 - lam) *
                    self.criterion(pred, target_b))
        else:
            return self.criterion(pred, target_a)
