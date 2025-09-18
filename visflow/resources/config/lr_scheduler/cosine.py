from __future__ import annotations

import pydantic as pydt


class CosineConfig(pydt.BaseModel):
    t_max: int = pydt.Field(10, description="Maximum number of iterations.", ge=1)

    eta_min: float = pydt.Field(0.0, description="Minimum learning rate.", ge=0.0)
