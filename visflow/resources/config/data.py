from __future__ import annotations

import pydantic as pydt


class DataConfig(pydt.BaseModel):
    train_dir: str = pydt.Field(
        default="./data/train", description="Path to the training dataset directory."
    )

    val_dir: str = pydt.Field(
        default="./data/val", description="Path to the validation dataset directory."
    )

    test_dir: str = pydt.Field(
        default="./data/test", description="Path to the test dataset directory."
    )

    num_workers: int = pydt.Field(
        default=4,
        ge=0,
        le=32,
        description="Number of worker processes for data loading.",
    )

    pin_memory: bool = pydt.Field(
        default=True, description="Whether to pin memory in data loaders."
    )
