from __future__ import annotations

import typing as t

from visflow.helper.settings import TypeBase
from visflow.helper.settings.fields import Field


class StepConfig(TypeBase):
    type: t.Literal["step"] = Field(
        default="step",
        description="Discriminator for Step Learning Rate Scheduler",
    )

    step_size: int = Field(
        default=10,
        description="Period of learning rate decay",
        ge=1,
        lt=1_000_000,
    )

    gamma: float = Field(
        default=0.1,
        description="Multiplicative factor of learning rate decay",
        gt=0.0,
        lt=1.0,
    )
