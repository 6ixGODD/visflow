from __future__ import annotations

import pydantic as pydt


class StepConfig(pydt.BaseModel):
    step_size: int = pydt.Field(
        10,
        description="Period of learning rate decay",
        ge=1,
    )

    gamma: float = pydt.Field(
        0.1,
        description="Multiplicative factor of learning rate decay",
        gt=0.0,
        lt=1.0,
    )
