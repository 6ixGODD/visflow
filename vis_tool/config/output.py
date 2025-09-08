from __future__ import annotations

import pydantic as pydt


class OutputConfig(pydt.BaseModel):
    """Output and logging configuration."""

    save_dir: str = pydt.Field(
        default='./output',
        description="Directory where training outputs will be saved."
    )

    experiment_name: str = pydt.Field(
        default='default_experiment',
        min_length=1,
        description="Unique name for the experiment."
    )

    save_plots: bool = pydt.Field(
        default=True,
        description="Whether to generate and save training plots."
    )

    save_model_every_n_epochs: int = pydt.Field(
        default=1,
        ge=1,
        description="Frequency of model checkpoint saving."
    )
