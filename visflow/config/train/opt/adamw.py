from __future__ import annotations

import typing as t

from visflow.helper.settings import Field
from visflow.helper.settings import TypeBase


class AdamWConfig(TypeBase):
    type: t.Literal["adamw"] = Field(
        default="adamw",
        description="Discriminator for optimizer type. Fixed to 'adamw' for AdamW optimizer.",
    )

    learning_rate: float = Field(
        default=0.001,
        ge=1e-6,
        le=1.0,
        description="Learning rate for the AdamW optimizer.",
    )

    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay (L2 regularization) factor.",
    )

    beta1: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Exponential decay rate for the first moment estimates.",
    )

    beta2: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Exponential decay rate for the second moment estimates.",
    )

    epsilon: float = Field(
        default=1e-8,
        ge=1e-10,
        le=1.0,
        description="Small constant for numerical stability.",
    )
