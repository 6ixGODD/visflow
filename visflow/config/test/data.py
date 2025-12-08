from __future__ import annotations

import os

from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class DataConfig(BaseModel):
    test_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Number of samples processed in each training batch.",
    )

    test_dir: os.PathLike[str] = Field(
        default="data/test",
        description="Directory where the dataset is stored.",
    )

    num_workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description="Number of worker processes for data loading.",
    )

    pin_memory: bool = Field(
        default=True,
        description="Whether to pin memory in data loaders.",
    )
