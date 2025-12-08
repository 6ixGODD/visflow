from __future__ import annotations

from visflow.config.shared.model import ModelConfig
from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class SharedConfig(BaseModel):
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Configuration for the machine learning model used in VisFlow.",
    )
