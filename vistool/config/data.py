from __future__ import annotations

import pydantic as pydt


class DataConfig(pydt.BaseModel):
    train_data_path: str = pydt.Field(
        default='./data/train',
        description="Path to the training dataset directory."
    )

    val_data_path: str = pydt.Field(
        default='./data/val',
        description="Path to the validation dataset directory."
    )

    test_data_path: str = pydt.Field(
        default='./data/test',
        description="Path to the test dataset directory."
    )

    input_size: int = pydt.Field(
        default=224,
        ge=32,
        le=1024,
        description="Input image size (height and width)."
    )

    num_workers: int = pydt.Field(
        default=4,
        ge=0,
        le=32,
        description="Number of worker processes for data loading."
    )

    pin_memory: bool = pydt.Field(
        default=True,
        description="Whether to pin memory in data loaders."
    )
