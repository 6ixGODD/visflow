from __future__ import annotations

import typing as t

from visflow.helper.settings import EnabledBase
from visflow.helper.settings import Field


class EarlyStoppingConfig(EnabledBase):
    enabled: bool = Field(
        default=True, description="Whether to enable early stopping during training."
    )

    patience: int = Field(
        default=5,
        description="Number of epochs with no improvement after which training will be stopped.",
    )

    min_delta: float = Field(
        default=0.0,
        description="Minimum change in the monitored metric to qualify as an improvement.",
    )

    target: t.Literal["loss", "accuracy", "f1", "precision", "recall"] = Field(
        default="loss", description="The metric to monitor for early stopping."
    )
