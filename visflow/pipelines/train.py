from __future__ import annotations

import typing as t

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from visflow.configs import TrainConfig
from visflow.logger import _Logger
from visflow.models import BaseClassifier, create_model


class TrainPipeline:
    config: TrainConfig
    logger: _Logger
    model: BaseClassifier
    input_size: tuple[int, int]
    train_transforms: transforms.Compose
    val_transforms: transforms.Compose

    def __init__(self, config: TrainConfig):
        self.config = config
        if config.logging.backend == 'native':
            from visflow.logger.native import NativeLoggingBackend
            backend = NativeLoggingBackend()
        elif config.logging.backend == 'loguru':
            from visflow.logger.loguru import LoguruBackend
            backend = LoguruBackend()
        else:
            raise ValueError(
                f"Unsupported logging backend: {config.logging.backend}"
            )
        self.logger = _Logger(
            backend=backend,
            targets=config.logging.targets,
            initial_ctx=config.logging.extra_context
        )
        self.model = create_model(
            name=config.model.architecture,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            weights_path=config.model.weights_path
        )

        # Define transforms ----------------------------------------------------
        if isinstance(self.config.resize.size, int):  # square resize
            x, y = self.input_size = (self.config.data.input_size,
                                      self.config.data.input_size)
        else:  # tuple resize
            x, y = self.input_size = self.config.resize.size
        if self.config.augmentation.crop.enabled:  # adjust for crop margin
            if isinstance(self.config.augmentation.crop.margin, int):
                x += self.config.augmentation.crop.margin
                y += self.config.augmentation.crop.margin
            else:
                x += self.config.augmentation.crop.margin[0]
                y += self.config.augmentation.crop.margin[1]

        train_transforms: t.List[t.Callable] = [
            transforms.Resize(
                size=(x, y),
                interpolation=InterpolationMode(
                    self.config.resize.interpolation
                ),
                max_size=self.config.resize.max_size,
                antialias=self.config.resize.antialias
            )
        ]

        if self.config.augmentation.crop.enabled:
            train_transforms.append(
                transforms.RandomCrop(
                    size=self.input_size,
                    pad_if_needed=True,
                    fill=self.config.augmentation.crop.fill,
                    padding_mode=self.config.augmentation.crop.padding_mode
                )
            )

        if self.config.augmentation.horizontal_flip.enabled:
            train_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=self.config.augmentation.horizontal_flip.p
                )
            )

        if self.config.augmentation.rotation.enabled:
            train_transforms.append(
                transforms.RandomRotation(
                    degrees=self.config.augmentation.rotation.degrees,
                    interpolation=InterpolationMode(
                        self.config.augmentation.rotation.interpolation
                    ),
                    expand=self.config.augmentation.rotation.expand,
                    center=self.config.augmentation.rotation.center,
                    fill=self.config.augmentation.rotation.fill
                )
            )

        if self.config.augmentation.color_jitter.enabled:
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=self.config.augmentation.color_jitter.brightness,
                    contrast=self.config.augmentation.color_jitter.contrast,
                    saturation=self.config.augmentation.color_jitter.saturation,
                    hue=self.config.augmentation.color_jitter.hue
                )
            )

        if self.config.augmentation.affine.enabled:
            train_transforms.append(
                transforms.RandomAffine(
                    degrees=self.config.augmentation.affine.degrees,
                    translate=self.config.augmentation.affine.translate,
                    scale=self.config.augmentation.affine.scale,
                    shear=self.config.augmentation.affine.shear,
                    interpolation=InterpolationMode(
                        self.config.augmentation.affine.interpolation
                    ),
                    fill=self.config.augmentation.affine.fill,
                    center=self.config.augmentation.affine.center
                )
            )

        # ToTensor should be the last transform before normalization
        train_transforms.append(transforms.ToTensor())

        if self.config.normalization.enabled:
            train_transforms.append(
                transforms.Normalize(
                    mean=self.config.normalization.mean,
                    std=self.config.normalization.std,
                    inplace=self.config.normalization.inplace
                )
            )

        if self.config.augmentation.erasing.enabled:
            train_transforms.append(
                transforms.RandomErasing(
                    p=self.config.augmentation.erasing.p,
                    scale=self.config.augmentation.erasing.scale,
                    ratio=self.config.augmentation.erasing.ratio,
                    value=self.config.augmentation.erasing.value,
                    inplace=self.config.augmentation.erasing.inplace
                )
            )

        # Compose all transforms
        self.train_transforms = transforms.Compose(train_transforms)

        val_transforms: t.List[t.Callable] = [
            transforms.Resize(
                size=self.input_size,
                interpolation=InterpolationMode(
                    self.config.resize.interpolation
                ),
                max_size=self.config.resize.max_size,
                antialias=self.config.resize.antialias
            ),
            transforms.ToTensor()
        ]

        if self.config.normalization.enabled:
            val_transforms.append(
                transforms.Normalize(
                    mean=self.config.normalization.mean,
                    std=self.config.normalization.std,
                    inplace=self.config.normalization.inplace
                )
            )

        self.val_transforms = transforms.Compose(val_transforms)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
