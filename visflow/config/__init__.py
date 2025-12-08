from __future__ import annotations

from visflow.config.shared import SharedConfig
from visflow.config.test import TestConfig
from visflow.config.train import TrainConfig
from visflow.helper.settings import Field
from visflow.helper.settings import Settings


class Config(Settings):
    shared: SharedConfig = Field(
        default_factory=SharedConfig,
        description="Shared configuration",
    )

    train: TrainConfig = Field(
        default_factory=TrainConfig,
        description="Training configuration",
    )

    test: TestConfig = Field(
        default_factory=TestConfig,
        description="Testing configuration",
    )
