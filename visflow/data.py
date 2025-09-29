from __future__ import annotations

import typing as t

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode

from visflow.context import DatasetInfo
from visflow.resources.config import TrainConfig
from visflow.utils.functional import compute_class_weights


class Datamodule:

    def __init__(self, config: TrainConfig):
        self.config = config
        if isinstance(self.config.resize.size, int):  # square resize
            x, y = self.input_size = (self.config.resize.size, self.config.resize.size)
        else:  # tuple resize
            x, y = self.input_size = self.config.resize.size
        if self.config.augmentation.crop.enabled:  # adjust for crop margin
            if isinstance(self.config.augmentation.crop.margin, int):
                x += self.config.augmentation.crop.margin
                y += self.config.augmentation.crop.margin
            else:
                x += self.config.augmentation.crop.margin[0]
                y += self.config.augmentation.crop.margin[1]

        train_transforms: t.List[t.Callable[..., t.Any]] = [
            transforms.Resize(size=(x, y),
                              interpolation=InterpolationMode(self.config.resize.interpolation),
                              max_size=self.config.resize.max_size,
                              antialias=self.config.resize.antialias)
        ]

        if self.config.augmentation.crop.enabled:
            train_transforms.append(
                transforms.RandomCrop(size=self.input_size,
                                      pad_if_needed=True,
                                      fill=self.config.augmentation.crop.fill,
                                      padding_mode=self.config.augmentation.crop.padding_mode))

        if self.config.augmentation.horizontal_flip.enabled:
            train_transforms.append(
                transforms.RandomHorizontalFlip(p=self.config.augmentation.horizontal_flip.p))

        if self.config.augmentation.rotation.enabled:
            train_transforms.append(
                transforms.RandomRotation(degrees=self.config.augmentation.rotation.degrees,
                                          interpolation=InterpolationMode(
                                              self.config.augmentation.rotation.interpolation),
                                          expand=self.config.augmentation.rotation.expand,
                                          center=self.config.augmentation.rotation.center,
                                          fill=self.config.augmentation.rotation.fill))

        if self.config.augmentation.color_jitter.enabled:
            train_transforms.append(
                transforms.ColorJitter(brightness=self.config.augmentation.color_jitter.brightness,
                                       contrast=self.config.augmentation.color_jitter.contrast,
                                       saturation=self.config.augmentation.color_jitter.saturation,
                                       hue=self.config.augmentation.color_jitter.hue))

        if self.config.augmentation.affine.enabled:
            train_transforms.append(
                transforms.RandomAffine(degrees=self.config.augmentation.affine.degrees,
                                        translate=self.config.augmentation.affine.translate,
                                        scale=self.config.augmentation.affine.scale,
                                        shear=self.config.augmentation.affine.shear,
                                        interpolation=InterpolationMode(
                                            self.config.augmentation.affine.interpolation),
                                        fill=self.config.augmentation.affine.fill,
                                        center=self.config.augmentation.affine.center))

        # ToTensor should be the last transform before normalization
        train_transforms.append(transforms.ToTensor())

        if self.config.normalization.enabled:
            train_transforms.append(
                transforms.Normalize(mean=self.config.normalization.mean,
                                     std=self.config.normalization.std,
                                     inplace=self.config.normalization.inplace))

        if self.config.augmentation.erasing.enabled:
            train_transforms.append(
                transforms.RandomErasing(p=self.config.augmentation.erasing.p,
                                         scale=self.config.augmentation.erasing.scale,
                                         ratio=self.config.augmentation.erasing.ratio,
                                         value=self.config.augmentation.erasing.value,
                                         inplace=self.config.augmentation.erasing.inplace))

        # Compose all transforms
        self.train_transforms = transforms.Compose(train_transforms)

        val_transforms: t.List[t.Callable[..., torch.Tensor]] = [
            transforms.Resize(size=self.input_size,
                              interpolation=InterpolationMode(self.config.resize.interpolation),
                              max_size=self.config.resize.max_size,
                              antialias=self.config.resize.antialias),
            transforms.ToTensor()
        ]

        if self.config.normalization.enabled:
            val_transforms.append(
                transforms.Normalize(mean=self.config.normalization.mean,
                                     std=self.config.normalization.std,
                                     inplace=self.config.normalization.inplace))

        self.val_transforms = transforms.Compose(val_transforms)

        self.train_set = ImageFolder(root=self.config.data.train_dir,
                                     transform=self.train_transforms)
        self.val_set = ImageFolder(root=self.config.data.val_dir, transform=self.val_transforms)
        self.test_set = ImageFolder(root=self.config.data.test_dir, transform=self.val_transforms)
        if (self.train_set.classes != self.val_set.classes
                or self.train_set.classes != self.test_set.classes):
            raise ValueError("Train, validation, and test sets must have the same classes")

    @property
    def train_loader(self) -> DataLoader[torch.Tensor]:
        if self.config.training.weighted_sampling:
            labels = [label for _, label in self.train_set.samples]
            class_weights = compute_class_weights(labels)
            sample_weights = [class_weights[label] for _, label in self.train_set.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = None
        return DataLoader(dataset=self.train_set,
                          batch_size=self.config.training.batch_size,
                          shuffle=self.config.training.shuffle,
                          num_workers=self.config.data.num_workers,
                          pin_memory=self.config.data.pin_memory,
                          drop_last=self.config.training.drop_last,
                          sampler=sampler)

    @property
    def val_loader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(dataset=self.val_set,
                          batch_size=self.config.training.batch_size,
                          shuffle=False,
                          num_workers=self.config.data.num_workers,
                          pin_memory=self.config.data.pin_memory,
                          drop_last=False)

    @property
    def test_loader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(dataset=self.test_set,
                          batch_size=self.config.testing.batch_size,
                          shuffle=False,
                          num_workers=self.config.data.num_workers,
                          pin_memory=self.config.data.pin_memory,
                          drop_last=False)

    @property
    def loaders(
        self
    ) -> t.Tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
        return self.train_loader, self.val_loader, self.test_loader

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(num_classes=len(self.train_set.classes),
                           train_size=len(self.train_set),
                           val_size=len(self.val_set),
                           test_size=len(self.test_set),
                           classes=self.train_set.classes)

    @property
    def classes(self) -> t.List[str]:
        return self.train_set.classes

    @property
    def class_to_idx(self) -> t.Dict[str, int]:
        return self.train_set.class_to_idx
