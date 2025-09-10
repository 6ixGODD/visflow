from __future__ import annotations

import pydantic as pydt


class OutputConfig(pydt.BaseModel):
    output_dir: str = pydt.Field(
        default='./output',
        description="Directory where training outputs will be saved."
    )

    experiment_name: str = pydt.Field(
        default='exp',
        min_length=1,
        description="Unique name for the experiment."
    )

