from __future__ import annotations

import abc
import logging
import os
import pathlib as p
import typing as t

import torch
import torch.nn as nn
import torchvision.models as models

from visflow.types import Checkpoint

logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module, abc.ABC):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        self._num_classes = value

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def load(
        self,
        ckpt: Checkpoint,
        *,
        strict: bool = True,
        map_location: str | torch.device | None = None
    ) -> None:
        classes = ckpt.get('classes', [])
        self.num_classes = len(classes) or self.num_classes
        self.load_state_dict(ckpt['model_state_dict'], strict=strict)

    def loads(
        self,
        ckpt_path: str | os.PathLike[str],
        *,
        strict: bool = True,
        map_location: str | torch.device | None = None
    ) -> None:
        path = p.Path(ckpt_path)
        ckpt: Checkpoint = torch.load(path, map_location=map_location or "cpu")
        self.load(ckpt, strict=strict, map_location=map_location)

    def last_conv(self) -> nn.Module:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement last_conv method."
        )


MODEL_REGISTRY: t.Dict[str, t.Callable[..., BaseClassifier]] = {}

_C = t.TypeVar("_C", bound=BaseClassifier)


def register_model(name: str) -> t.Callable[[t.Type[_C]], t.Type[_C]]:
    def decorator(cls: t.Type[_C]) -> t.Type[_C]:
        if not issubclass(cls, BaseClassifier):
            raise TypeError(f"{cls.__name__} must inherit from BaseClassifier")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class TorchVisionClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        weights_path: str | os.PathLike[str] | None = None,
        map_location: str | torch.device | None = None,
        **kwargs: t.Any
    ) -> None:
        super().__init__(num_classes=num_classes)
        if not hasattr(models, model_name):
            raise ValueError(
                f"'{model_name}' not found in torchvision.models. "
            )

        fn = getattr(models, model_name, None)
        if not callable(fn):
            raise ValueError(f"'{model_name}' is not callable.")

        if weights_path:
            model = fn(weights=None, **kwargs)
            self.backbone: nn.Module = model
            self._replace_head(num_classes)
            state_dict = torch.load(
                weights_path,
                map_location=map_location or "cpu"
            )
            missing, unexpected = self.backbone.load_state_dict(
                state_dict,
                strict=False
            )
            if missing:
                logger.debug(f"[WARN] Missing keys: {missing}")
            if unexpected:
                logger.debug(f"[WARN] Unexpected keys: {unexpected}")
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

        elif isinstance(m, models.VGG):
            num_features = m.classifier[6].in_features
            m.classifier[6] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.SqueezeNet):
            m.classifier[1] = nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            m.num_classes = num_classes

        elif isinstance(m, models.DenseNet):
            num_features = m.classifier.in_features
            m.classifier = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.Inception3):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MobileNetV2):
            num_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MobileNetV3):
            num_features = m.classifier[3].in_features
            m.classifier[3] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.EfficientNet):
            num_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.ConvNeXt):
            num_features = m.classifier[2].in_features
            m.classifier[2] = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.GoogLeNet):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.RegNet):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.ShuffleNetV2):
            num_features = m.fc.in_features
            m.fc = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.VisionTransformer):
            num_features = m.heads.head.in_features
            m.heads.head = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.SwinTransformer):
            num_features = m.head.in_features
            m.head = nn.Linear(num_features, num_classes)

        elif isinstance(m, models.MaxVit):
            num_features = m.head.in_features
            m.head = nn.Linear(num_features, num_classes)

        else:
            raise NotImplementedError(
                f"Head replacement not implemented for {type(m).__name__}. "
            )

    def last_conv(self) -> nn.Module:
        m = self.backbone

        if isinstance(m, models.ResNet):
            block = m.layer4[-1]
            if hasattr(block, "conv3"):  # Bottleneck
                return block.conv3
            else:  # BasicBlock
                return block.conv2

        elif isinstance(m, models.VGG):
            return m.features[-1]

        elif isinstance(m, models.SqueezeNet):
            return m.features[-1]

        elif isinstance(m, models.DenseNet):
            block = m.features[-1]
            return block[-1].conv2

        elif isinstance(m, models.Inception3):
            return m.Mixed_7c.branch3[0]

        elif isinstance(m, models.MobileNetV2):
            block = m.features[-1]
            return block.conv[-1]

        elif isinstance(m, models.MobileNetV3):
            return m.features[-1][0]

        elif isinstance(m, models.EfficientNet):
            return m.features[-1][0]

        elif isinstance(m, models.ConvNeXt):
            return m.stages[-1][-1].dwconv

        elif isinstance(m, models.GoogLeNet):
            return m.inception5b.branch3[0]

        elif isinstance(m, models.RegNet):
            block = m.trunk_output[-1]
            return block[-1]

        elif isinstance(m, models.ShuffleNetV2):
            return m.stage4[-1].conv2

        elif isinstance(m, models.VisionTransformer):
            raise NotImplementedError(
                "last_conv is not defined for VisionTransformer (no conv "
                "layer)."
            )

        elif isinstance(m, models.SwinTransformer):
            raise NotImplementedError(
                "last_conv is not defined for SwinTransformer (no conv layer)."
            )

        elif isinstance(m, models.MaxVit):
            raise NotImplementedError(
                "last_conv is not defined for MaxVit (no conv layer)."
            )

        else:
            raise NotImplementedError(
                f"last_conv not implemented for {type(m).__name__}. "
                f"Please check the model architecture."
            )


def make_model(
    name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
    weights_path: str | os.PathLike[str] | None = None,
    **kwargs: t.Any
) -> BaseClassifier:
    if hasattr(models, name):  # torchvision
        return TorchVisionClassifier(
            model_name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            weights_path=weights_path,
            **kwargs
        )

    if name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
        if weights_path:
            model.loads(weights_path)
        return model

    raise ValueError(f"Unknown model: {name}")


def load_model(
    ckpt_path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
    **kwargs: t.Any
) -> BaseClassifier:
    path = p.Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return load_model_from_ckpt(
        torch.load(path, map_location=map_location or "cpu"),
        strict=strict,
        map_location=map_location,
        **kwargs
    )


def load_model_from_ckpt(
    ckpt: Checkpoint,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
    **kwargs: t.Any
) -> BaseClassifier:
    model_name = ckpt.get('config', {}).get('model', {}).get('architecture', '')
    if not model_name:
        raise ValueError("Checkpoint does not contain 'model_name'.")

    num_classes = len(ckpt.get('classes', [])) or kwargs.pop('num_classes', 2)
    model = make_model(
        model_name,
        num_classes=num_classes,
        **kwargs
    )
    model.load(ckpt, strict=strict, map_location=map_location)
    return model
