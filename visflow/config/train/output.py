from __future__ import annotations

import os
import typing as t

from visflow.config.train.logging import LoggingConfig
from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class OutputConfig(BaseModel):
    output_dir: os.PathLike[str] = Field(
        default="./output",
        description="Directory where training outputs will be saved.",
    )

    experiment_name: t.Literal["auto"] | str = Field(
        default="auto",
        min_length=1,
        description="Unique name for the experiment. Use 'auto' to generate a name based on model architecture.",
    )

    checkpoint_frequency: int = Field(
        default=10,
        ge=1,
        description="Frequency of model checkpoint saving.",
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration for training output.",
    )
