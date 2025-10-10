from __future__ import annotations

import pydantic as pydt


class TestingConfig(pydt.BaseModel):
    batch_size: int = pydt.Field(
        default=32,
        ge=1,
        le=512,
        description="Number of samples processed in each training batch.",
    )
