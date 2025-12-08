from __future__ import annotations

import os

from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class DataConfig(BaseModel):
    train_batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for training data loaders.",
    )

    train_shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the training data.",
    )

    train_dir: os.PathLike[str] = Field(
        default="data/train",
        description="Directory path for training data.",
    )

    val_batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        description="Batch size for validation data loaders.",
    )

    val_shuffle: bool = Field(
        default=False,
        description="Whether to shuffle the validation data.",
    )

    val_dir: os.PathLike[str] = Field(
        default="data/val",
        description="Directory path for validation data.",
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

    weighted_sampling: bool = Field(
        default=False,
        description="Whether to use weighted sampling to handle class imbalance.",
    )

    drop_last: bool = Field(
        default=False,
        description="Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size.",
    )

    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Initial learning rate for optimization.",
    )
