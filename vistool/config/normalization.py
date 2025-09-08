from __future__ import annotations

import typing as t

import pydantic as pydt


class NormalizationConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=True,
        description="Whether to apply normalization to input images."
    )

    mean: t.List[float] = pydt.Field(
        default=[0.485, 0.456, 0.406],
        description="Mean values for normalization (RGB channels). ImageNet "
                    "default: [0.485, 0.456, 0.406]"
    )

    std: t.List[float] = pydt.Field(
        default=[0.229, 0.224, 0.225],
        description="Standard deviation values for normalization (RGB "
                    "channels). ImageNet default: [0.229, 0.224, 0.225]"
    )

    @pydt.field_validator('mean', 'std')
    @classmethod
    def validate_channels(cls, v):
        if len(v) != 3:
            raise ValueError(
                "Mean and std must have exactly 3 values for RGB channels"
            )
        return v
