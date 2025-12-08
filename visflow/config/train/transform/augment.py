from __future__ import annotations

import typing as t

from visflow.helper.settings import BaseModel
from visflow.helper.settings import EnabledBase
from visflow.helper.settings import Field
from visflow.types import PixelValue


class CropConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random cropping.",
    )

    margin: int | tuple[int, int] = Field(
        default=0,
        ge=0,
        description="Margin to be left on each border of the image before cropping. If a single int is provided, it is used for all borders. If a tuple of two ints is provided, it is used for left/right and top/bottom respectively.",
        examples=[0, 10, (5, 15)],
    )

    fill: PixelValue = Field(
        default=0,
        description="Pixel fill value for the area outside the image if padding is applied. If a single int is provided, it is used for all channels. If a tuple of three ints is provided, it is used for the R, G, B channels respectively.",
        examples=[0, (255, 0, 0), (0.0, 255.0, 0.0)],
    )

    padding_mode: t.Literal["constant", "edge", "reflect", "symmetric"] = Field(
        default="constant",
        description="Padding mode to use if padding is applied.",
    )


class HorizontalFlipConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random horizontal flip.",
    )

    p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying horizontal flip.",
    )


class RotationConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random rotation.",
    )

    degrees: float | tuple[float, float] = Field(
        default=15.0,
        description="Range of degrees to select from. If degrees is a number, the range will be (-degrees, +degrees).",
        examples=[15.0, (-30.0, 30.0)],
    )

    interpolation: t.Literal["nearest", "nearest-exact", "bilinear", "bicubic"] = Field(
        default="nearest",
        description="Interpolation method for rotation.",
    )

    expand: bool = Field(
        default=False,
        description="Whether to expand the image to make it large enough to hold the entire rotated image.",
    )

    center: tuple[float, float] | None = Field(
        default=None,
        description="Optional center of rotation. Origin is the upper left corner.",
    )

    fill: PixelValue = Field(
        default=0,
        description="Pixel fill value for the area outside the rotated image.",
        examples=[0, (255, 0, 0), (0.0, 255.0, 0.0)],
    )


class ColorJitterConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random color jittering.",
    )

    brightness: float | tuple[float, float] = Field(
        default=0.2,
        description="How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].",
    )

    contrast: float | tuple[float, float] = Field(
        default=0.2,
        description="How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].",
    )

    saturation: float | tuple[float, float] = Field(
        default=0.2,
        description="How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].",
        examples=[0.2, (0.5, 1.5)],
    )

    hue: float | tuple[float, float] = Field(
        default=0.1,
        description="How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue].",
        examples=[0.1, (-0.2, 0.2)],
    )


class AffineConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random affine transformations.",
    )

    degrees: float | tuple[float, float] = Field(
        default=0.0,
        description="Range of degrees to select from for rotation.",
        examples=[15.0, (-30.0, 30.0)],
    )

    translate: tuple[float, float] | None = Field(
        default=None,
        description="Tuple of maximum absolute fraction for horizontal and vertical translations.",
        examples=[(0.1, 0.1), (0.2, 0.3)],
    )

    scale: tuple[float, float] | None = Field(
        default=None,
        description="Scaling factor interval.",
        examples=[(0.8, 1.2), (0.5, 1.5)],
    )

    shear: float | tuple[float, float] | None = Field(
        default=None,
        description="Range of degrees to select from for shearing.",
        examples=[10.0, (-20.0, 20.0)],
    )

    interpolation: t.Literal["nearest", "bilinear", "bicubic"] = Field(
        default="bilinear",
        description="Interpolation method for affine transformation.",
    )

    fill: PixelValue = Field(
        default=0.0,
        description="Pixel fill value for the area outside the transformed image.",
        examples=[0, (255, 0, 0), (0.0, 255.0, 0.0)],
    )

    center: tuple[float, float] | None = Field(
        default=None,
        description="Optional center of rotation. Origin is the upper left corner.",
        examples=[(50.0, 50.0), (100.0, 150.0)],
    )


class ErasingConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply random erasing.",
    )

    p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability that the Random Erasing operation will be performed.",
    )

    scale: tuple[float, float] = Field(
        default=(0.02, 0.33),
        description="Range of proportion of erased area against input image.",
    )

    ratio: tuple[float, float] = Field(
        default=(0.3, 3.3),
        description="Range of aspect ratio of erased area.",
    )

    value: PixelValue | t.Literal["random"] = Field(
        default=0,
        description="Pixel value for erased area. If a single int is provided, it is used for all channels. If a tuple of three ints is provided, it is used for the R, G, B channels respectively. If 'random', erasing value is random for each pixel.",
        examples=[0, (255, 0, 0), (0.0, 255.0, 0.0), "random"],
    )

    inplace: bool = Field(
        default=False,
        description="Whether to make this operation in-place.",
    )


class MixupConfig(EnabledBase):
    enabled: bool = Field(
        default=False,
        description="Whether to apply MixUp data augmentation.",
    )

    alpha: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="MixUp interpolation strength parameter.",
    )

    p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying MixUp.",
    )


class AugmentationConfig(BaseModel):
    crop: CropConfig = Field(
        default_factory=CropConfig,
        description="Random crop configuration.",
    )

    horizontal_flip: HorizontalFlipConfig = Field(
        default_factory=HorizontalFlipConfig,
        description="Horizontal flip configuration.",
    )

    rotation: RotationConfig = Field(
        default_factory=RotationConfig,
        description="Rotation configuration.",
    )

    color_jitter: ColorJitterConfig = Field(
        default_factory=ColorJitterConfig,
        description="Color jitter configuration.",
    )

    affine: AffineConfig = Field(
        default_factory=AffineConfig,
        description="Affine transformation configuration.",
    )

    erasing: ErasingConfig = Field(
        default_factory=ErasingConfig,
        description="Random erasing configuration.",
    )

    mixup: MixupConfig = Field(
        default_factory=MixupConfig,
        description="MixUp configuration.",
    )
