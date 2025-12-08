from __future__ import annotations

import os
import typing as t

import torch
import torch.nn as nn
import torchvision.models as models

from visflow.helper import LoggingMixin
from visflow.modules import VFModel


class TorchvisionWrapper(LoggingMixin, VFModel):
    """Wrapper for torchvision models with unified interface.

    This class wraps torchvision models and provides automatic head replacement
    for different number of classes, along with GradCAM layer identification.

    Args:
        model_name: Name of the torchvision model.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        weights_path: Path to custom weights file.
        map_location: Device location for loading weights.
        **kwargs: Additional arguments for model creation.

    Raises:
        ValueError: If model_name is not found in torchvision.models or not
            callable.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        model_name: str,
        pretrained: bool = True,
        weights_path: str | os.PathLike[str] | None = None,
        map_location: str | torch.device | None = None,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if not hasattr(models, model_name):
            raise ValueError(f"'{model_name}' not found in torchvision.models. ")

        fn = t.cast(t.Callable[..., nn.Module], getattr(models, model_name, None))
        if not callable(fn):
            raise ValueError(f"'{model_name}' is not callable.")
        if weights_path:
            model = fn(weights=None, **kwargs)  # type: nn.Module
            self.backbone = model
            self._replace_head(num_classes)
            state_dict = torch.load(weights_path, map_location=map_location or "cpu")
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                self.logger.warning(f"Missing keys: {missing}")
            if unexpected:
                self.logger.warning(f"Unexpected keys: {unexpected}")
        else:
            model = fn(weights="DEFAULT" if pretrained else None, **kwargs)
            self.backbone = model
            self._replace_head(num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        if value != self._num_classes:
            self._replace_head(value)
            self._num_classes = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return t.cast(torch.Tensor, self.backbone(x))

    def _replace_head(self, num_classes: int) -> None:
        m = self.backbone

        if isinstance(m, models.ResNet):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, (models.AlexNet, models.VGG)):
            num_features = m.classifier[6].in_features  # type: ignore[attr-defined]
            m.classifier[6] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.SqueezeNet):
            m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            m.num_classes = num_classes

        elif isinstance(m, models.DenseNet):
            num_features = m.classifier.in_features
            m.classifier = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.Inception3):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MobileNetV2):
            num_features = m.classifier[1].in_features  # type: ignore[attr-defined]
            m.classifier[1] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MobileNetV3):
            num_features = m.classifier[3].in_features  # type: ignore[attr-defined]
            m.classifier[3] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.EfficientNet):
            num_features = m.classifier[1].in_features  # type: ignore[attr-defined]
            m.classifier[1] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.ConvNeXt):
            num_features = m.classifier[2].in_features  # type: ignore[attr-defined]
            m.classifier[2] = nn.Linear(num_features, num_classes)

        elif isinstance(m, (models.GoogLeNet, models.RegNet, models.ShuffleNetV2)):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.VisionTransformer):
            num_features = m.heads.head.in_features
            m.heads.head = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.SwinTransformer):
            num_features = m.head.in_features
            m.head = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MaxVit):
            num_features = m.classifier[-1].in_features  # type: ignore[attr-defined]
            m.classifier[-1] = nn.Linear(num_features, num_classes, bias=False)

        else:
            raise NotImplementedError(f"Head replacement not implemented for {type(m).__name__}. ")

    @property
    def gradcam_layer(self) -> nn.Module:
        m = self.backbone

        # fmt: off
        if isinstance(m, models.ResNet):
            from torchvision.models.resnet import BasicBlock
            from torchvision.models.resnet import Bottleneck
            if isinstance(block := m.layer4[-1], Bottleneck):  # Bottleneck
                return block.conv3  # type: ignore[return-value]
            if isinstance(block := m.layer4[-1], BasicBlock):  # BasicBlock
                return block.conv2  # type: ignore[return-value]
            raise NotImplementedError("`gradcam_layer` not implemented for this ResNet block type.")

        if isinstance(m, models.AlexNet):
            for layer in reversed(m.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
            raise NotImplementedError("No Conv2d layer found in AlexNet features.")

        if isinstance(m, models.VGG):
            for layer in reversed(t.cast(nn.Sequential, m.features)):
                if isinstance(layer, nn.Conv2d):
                    return layer
            raise NotImplementedError("No Conv2d layer found in VGG features.")

        if isinstance(m, models.SqueezeNet):
            from torchvision.models.squeezenet import Fire
            if isinstance(block := m.features[-1], Fire):
                return block  # type: ignore[return-value]
            raise NotImplementedError("gradcam_layer not implemented for this SqueezeNet block type.")

        if isinstance(m, models.DenseNet):
            from torchvision.models.densenet import _DenseLayer
            if isinstance(block := m.features[-2][-1], _DenseLayer):
                return block.conv2  # type: ignore[return-value]
            raise NotImplementedError("gradcam_layer not implemented for this DenseNet block type.")

        if isinstance(m, models.Inception3):
            return m.Mixed_7c  # type: ignore[return-value]

        if isinstance(m, (models.MobileNetV2, models.MobileNetV3)):
            import torchvision.ops as ops
            if isinstance(block := m.features[-1], ops.Conv2dNormActivation):
                return block[0]  # type: ignore[return-value]
            raise NotImplementedError("gradcam_layer not implemented for this MobileNet block type.")

        if isinstance(m, models.EfficientNet):
            import torchvision.ops as ops
            if isinstance(block := m.features[-1], ops.Conv2dNormActivation):
                return block[0]  # type: ignore[return-value]
            raise NotImplementedError("gradcam_layer not implemented for this EfficientNet block type.")

        if isinstance(m, models.ConvNeXt):
            from torchvision.models.convnext import CNBlock
            if isinstance(block := m.features[-1][-1], CNBlock):
                return t.cast(nn.Conv2d, block.block[0])
            raise NotImplementedError("gradcam_layer not implemented for this ConvNeXt block type.")

        if isinstance(m, models.GoogLeNet):
            return m.inception5b  # type: ignore[return-value]

        if isinstance(m, models.RegNet):
            from torchvision.models.regnet import AnyStage
            from torchvision.models.regnet import ResBottleneckBlock
            if isinstance(stage := m.trunk_output[-1], AnyStage) and isinstance(block := stage[-1], ResBottleneckBlock):
                return block.f.c  # type: ignore[return-value]
            raise NotImplementedError("gradcam_layer not implemented for this RegNet block type.")
        if isinstance(m, models.ShuffleNetV2):
            return m.stage4  # type: ignore[return-value]

        if isinstance(m, models.VisionTransformer):
            raise NotImplementedError("gradcam_layer is not defined for VisionTransformer (no conv layer).")

        if isinstance(m, models.SwinTransformer):
            raise NotImplementedError("gradcam_layer is not defined for SwinTransformer (no conv layer).")

        if isinstance(m, models.MaxVit):
            raise NotImplementedError("gradcam_layer is not defined for MaxVit (no conv layer).")

        raise NotImplementedError(
            f"gradcam_layer not implemented for {type(m).__name__}. Please check the model architecture."
        )
