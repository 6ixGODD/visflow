from __future__ import annotations

import typing as t

import pydantic as pydt


class NormalizationConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(default=True,
                               description="Whether to apply normalization to input images.")

    mean: t.Tuple[float, float, float] = pydt.Field(
        default=(0.485, 0.456, 0.406),
        description="Mean values for normalization (RGB channels). ImageNet "
        "default: (0.485, 0.456, 0.406)",
    )

    std: t.Tuple[float, float, float] = pydt.Field(
        default=(0.229, 0.224, 0.225),
        description="Standard deviation values for normalization (RGB "
        "channels). ImageNet default: (0.229, 0.224, 0.225)",
    )

    inplace: bool = pydt.Field(
        default=False,
        description="Whether to perform normalization in-place to save memory.",
    )
