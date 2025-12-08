from __future__ import annotations

import typing as t

from visflow.config.train.data import DataConfig
from visflow.config.train.early_stopping import EarlyStoppingConfig
from visflow.config.train.lr.cos import CosineConfig
from visflow.config.train.lr.plateau import PlateauConfig
from visflow.config.train.lr.step import StepConfig
from visflow.config.train.opt.adam import AdamConfig
from visflow.config.train.opt.adamw import AdamWConfig
from visflow.config.train.opt.sgd import SGDConfig
from visflow.config.train.output import OutputConfig
from visflow.config.train.transform import TransformConfig
from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class TrainConfig(BaseModel):
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data loading configuration",
    )

    lr_scheduler: CosineConfig | PlateauConfig | StepConfig = Field(
        discriminator="type",
        default_factory=CosineConfig,
        description="Learning rate scheduler configuration",
        examples=[CosineConfig(), PlateauConfig(), StepConfig()],
    )

    transform: TransformConfig = Field(
        default_factory=TransformConfig,
        description="Data transformation configuration",
    )

    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Training output configuration",
    )

    device: t.Literal["cpu", "cuda"] | str = Field(
        default="cuda",
        description="Device to use for training. Can be 'cpu', 'cuda', or a specific device identifier like 'cuda:0'.",
        examples=["cpu", "cuda", "cuda:0"],
    )

    epochs: int = Field(
        default=10,
        ge=1,
        description="Maximum number of training epochs.",
    )

    optimizer: SGDConfig | AdamConfig | AdamWConfig = Field(
        discriminator="type",
        default_factory=SGDConfig,
        description="Optimizer configuration",
        examples=[SGDConfig(), AdamConfig(), AdamWConfig()],
    )

    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description="Early stopping configuration to prevent overfitting.",
    )

    label_smoothing: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Label smoothing factor to prevent overconfident predictions.",
    )
