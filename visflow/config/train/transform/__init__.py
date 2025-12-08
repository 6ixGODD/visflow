from __future__ import annotations

from visflow.config.train.transform.augment import AugmentationConfig
from visflow.config.train.transform.norm import NormalizationConfig
from visflow.config.train.transform.resize import ResizeConfig
from visflow.helper.settings import BaseModel
from visflow.helper.settings import Field


class TransformConfig(BaseModel):
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Configuration for data augmentation during training",
    )

    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description="Configuration for data normalization",
    )

    resize: ResizeConfig = Field(
        default_factory=ResizeConfig,
        description="Configuration for resizing images",
    )
