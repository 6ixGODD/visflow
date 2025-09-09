from __future__ import annotations

import typing as t

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from visflow.configs import TrainConfig
from visflow.logging import Logger
from visflow.models import make_model



class TrainPipeline:
    def __init__(self, config: TrainConfig, logger: Logger):
        self.config = config
        self.logger = logger

        self.model = make_model(
            name=config.model.architecture,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            weights_path=config.model.weights_path
        )

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
