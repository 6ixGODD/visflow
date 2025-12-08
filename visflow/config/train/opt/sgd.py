from __future__ import annotations

import typing as t

from visflow.helper.settings import Field
from visflow.helper.settings import TypeBase


class SGDConfig(TypeBase):
    type: t.Literal["sgd"] = Field(
        default="sgd",
        description="Discriminator for optimizer type. Fixed to 'sgd' for SGD optimizer.",
    )

    learning_rate: float = Field(
        default=0.001,
        ge=1e-6,
        le=1.0,
        description="Learning rate for the SGD optimizer.",
    )

    momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum factor for SGD optimizer.",
    )

    weight_decay: float = Field(
        default=0.0005,
        ge=0.0,
        description="Weight decay (L2 penalty) for SGD optimizer.",
    )

    nesterov: bool = Field(
        default=True,
        description="Whether to use Nesterov momentum.",
    )

    dampening: float = Field(
        default=0.0,
        ge=0.0,
        description="Dampening for momentum.",
    )
