from __future__ import annotations

import typing as t

import torch

from visflow.data import ImageDataModule
from visflow.resources.configs import TrainConfig
from visflow.resources.logger import Logger
from visflow.resources.models import make_model


class TrainPipeline:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.logger = Logger(config.logging)
        self.model = make_model(
            name=config.model.architecture,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            weights_path=config.model.weights_path
        )
        self.device = torch.device(config.training.device)
        self.data_module = ImageDataModule(config)
        (self.train_loader,
         self.val_loader,
         self.test_loader) = self.data_module.loaders

        self.best_acc = 0.0
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
