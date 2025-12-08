from __future__ import annotations

from visflow.config.test.data import DataConfig
from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class TestConfig(BaseModel):
    data: DataConfig = Field(
        default_factory=DataConfig, description="Configuration for test data generation"
    )
