from __future__ import annotations

import typing as t

from visflow.helper.settings import Field
from visflow.helper.settings import TypeBase


class CosineConfig(TypeBase):
    type: t.Literal["cosine"] = Field(
        default="cosine",
        description="Discriminator field to identify the learning rate scheduler type. Fixed to 'cosine'.",
    )

    t_max: int = Field(
        10,
        description="Maximum number of iterations.",
        ge=1,
    )

    eta_min: float = Field(
        0.0,
        description="Minimum learning rate.",
        ge=0.0,
    )
