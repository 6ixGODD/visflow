from __future__ import annotations

import typing as t

import pydantic as pydt


class CropConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random cropping."
    )

    scale: t.Tuple[float, float] = pydt.Field(
        default=(0.8, 1.0),
        description="Range of the random area of the crop."
    )

    ratio: t.Tuple[float, float] = pydt.Field(
        default=(3. / 4., 4. / 3.),
        description="Range of aspect ratio of the crop."
    )


class HorizontalFlipConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random horizontal flip."
    )

    probability: float = pydt.Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying horizontal flip."
    )


class RotationConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random rotation."
    )

    degrees: float | t.Tuple[float, float] = pydt.Field(
        default=15.0,
        description="Range of degrees to select from. If degrees is a number, "
                    "the range will be (-degrees, +degrees)."
    )

    interpolation: t.Literal['nearest', 'bilinear', 'bicubic'] = pydt.Field(
        default='bilinear',
        description="Interpolation method for rotation."
    )

    expand: bool = pydt.Field(
        default=False,
        description="Whether to expand the image to make it large enough to "
                    "hold the entire rotated image."
    )

    center: t.Tuple[float, float] | None = pydt.Field(
        default=None,
        description="Optional center of rotation. Origin is the upper left "
                    "corner."
    )

    fill: float | t.Tuple[float, ...] = pydt.Field(
        default=0.0,
        description="Pixel fill value for the area outside the rotated image."
    )


class ColorJitterConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random color jittering."
    )

    brightness: float | t.Tuple[float, float] = pydt.Field(
        default=0.2,
        description="How much to jitter brightness. brightness_factor is "
                    "chosen uniformly from [max(0, 1 - brightness), "
                    "1 + brightness]."
    )

    contrast: float | t.Tuple[float, float] = pydt.Field(
        default=0.2,
        description="How much to jitter contrast. contrast_factor is chosen "
                    "uniformly from [max(0, 1 - contrast), 1 + contrast]."
    )

    saturation: float | t.Tuple[float, float] = pydt.Field(
        default=0.2,
        description="How much to jitter saturation. saturation_factor is "
                    "chosen uniformly from [max(0, 1 - saturation), "
                    "1 + saturation]."
    )

    hue: float | t.Tuple[float, float] = pydt.Field(
        default=0.1,
        description="How much to jitter hue. hue_factor is chosen uniformly "
                    "from [-hue, hue]."
    )


class AffineConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random affine transformations."
    )

    degrees: float | t.Tuple[float, float] = pydt.Field(
        default=0.0,
        description="Range of degrees to select from for rotation."
    )

    translate: t.Tuple[float, float] | None = pydt.Field(
        default=None,
        description="Tuple of maximum absolute fraction for horizontal and "
                    "vertical translations."
    )

    scale: t.Tuple[float, float] | None = pydt.Field(
        default=None,
        description="Scaling factor interval."
    )

    shear: float | t.Tuple[float, float] | None = pydt.Field(
        default=None,
        description="Range of degrees to select from for shearing."
    )

    interpolation: t.Literal['nearest', 'bilinear', 'bicubic'] = pydt.Field(
        default='bilinear',
        description="Interpolation method for affine transformation."
    )

    fill: float | t.Tuple[float, ...] = pydt.Field(
        default=0.0,
        description="Pixel fill value for the area outside the transformed "
                    "image."
    )


class ErasingConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply random erasing."
    )

    probability: float = pydt.Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability that the Random Erasing operation will be "
                    "performed."
    )

    scale: t.Tuple[float, float] = pydt.Field(
        default=(0.02, 0.33),
        description="Range of proportion of erased area against input image."
    )

    ratio: t.Tuple[float, float] = pydt.Field(
        default=(0.3, 3.3),
        description="Range of aspect ratio of erased area."
    )

    value: float | str = pydt.Field(
        default=0,
        description="Erasing value. If value is a single int, it is used to "
                    "erase all pixels. If 'random', random values are used."
    )

    inplace: bool = pydt.Field(
        default=False,
        description="Whether to make this operation in-place."
    )


class MixupConfig(pydt.BaseModel):
    enabled: bool = pydt.Field(
        default=False,
        description="Whether to apply MixUp data augmentation."
    )

    alpha: float = pydt.Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="MixUp interpolation strength parameter."
    )

    probability: float = pydt.Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying MixUp."
    )


class AugmentationConfig(pydt.BaseModel):
    crop: CropConfig = pydt.Field(
        default_factory=CropConfig,
        description="Random crop configuration."
    )

    horizontal_flip: HorizontalFlipConfig = pydt.Field(
        default_factory=HorizontalFlipConfig,
        description="Horizontal flip configuration."
    )

    rotation: RotationConfig = pydt.Field(
        default_factory=RotationConfig,
        description="Rotation configuration."
    )

    color_jitter: ColorJitterConfig = pydt.Field(
        default_factory=ColorJitterConfig,
        description="Color jitter configuration."
    )

    affine: AffineConfig = pydt.Field(
        default_factory=AffineConfig,
        description="Affine transformation configuration."
    )

    erasing: ErasingConfig = pydt.Field(
        default_factory=ErasingConfig,
        description="Random erasing configuration."
    )

    mixup: MixupConfig = pydt.Field(
        default_factory=MixupConfig,
        description="MixUp configuration."
    )
