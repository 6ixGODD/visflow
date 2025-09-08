from __future__ import annotations

import typing as t

import pydantic as pydt


class ModelConfig(pydt.BaseModel):
    """Model architecture configuration."""

    architecture: t.Literal[
                      'resnet18', 'resnet34', 'resnet50', 'resnet101',
                      'resnet152',
                      'vgg11', 'vgg13', 'vgg16', 'vgg19',
                      'densenet121', 'densenet169', 'densenet201',
                      'densenet161',
                      'mobilenet_v2', 'mobilenet_v3_small',
                      'mobilenet_v3_large',
                      'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                      'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                      'efficientnet_b6', 'efficientnet_b7'
                  ] | str = pydt.Field(
        default='resnet18',
        description="Model architecture to use for training."
    )

    pretrained: bool = pydt.Field(
        default=True,
        description="Whether to use pre-trained weights from ImageNet."
    )

    num_classes: int = pydt.Field(
        default=2,
        ge=1,
        description="Number of output classes for classification."
    )
