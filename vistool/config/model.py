from __future__ import annotations

import typing as t

import pydantic as pydt


class ModelConfig(pydt.BaseModel):
    architecture: (
        t.Literal[
            'alexnet',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
            'wide_resnet50_2', 'wide_resnet101_2',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
            'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
            'efficientnet_b6', 'efficientnet_b7',
            'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
            'convnext_tiny', 'convnext_small', 'convnext_base',
            'convnext_large',
            'googlenet',
            'inception_v3',
            'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
            'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf',
            'regnet_y_32gf', 'regnet_y_128gf', 'regnet_x_400mf',
            'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
            'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
            'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
            'vgg19', 'vgg19_bn',
            'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
            'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
            'maxvit_t'] | str
    ) = pydt.Field(
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

    weights_path: str | None = pydt.Field(
        default=None,
        description="Path to custom weights file (optional)."
    )
