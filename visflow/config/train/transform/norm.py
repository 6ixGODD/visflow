from __future__ import annotations

from visflow.helper.settings import EnabledBase
from visflow.helper.settings import Field


class NormalizationConfig(EnabledBase):
    enabled: bool = Field(
        default=True,
        description="Whether to apply normalization to input images.",
    )

    mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406),
        description="Mean values for normalization (RGB channels). ImageNet default: (0.485, 0.456, 0.406)",
    )

    std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225),
        description="Standard deviation values for normalization (RGB channels). ImageNet default: (0.229, 0.224, 0.225)",
    )

    inplace: bool = Field(
        default=False,
        description="Whether to perform normalization in-place to save memory.",
    )
