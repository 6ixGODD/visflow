from __future__ import annotations

import torch
import torch.nn as nn

from visflow.resources.models import BaseClassifier, register_model


@register_model('simple_cnn')
class SimpleCNN(BaseClassifier):
    """A simple Convolutional Neural Network (CNN) for image classification."""

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.AdaptiveAvgPool2d((8, 8)))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(64 * 8 * 8, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
