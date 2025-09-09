from __future__ import annotations

import torch
import torch.nn as nn

from visflow.types import CriterionFunc


class MixUpLoss(nn.Module):
    def __init__(self, criterion: CriterionFunc):
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
