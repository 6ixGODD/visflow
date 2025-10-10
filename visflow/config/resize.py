from __future__ import annotations

import typing as t

import pydantic as pydt


class ResizeConfig(pydt.BaseModel):
    size: int | t.Tuple[int, int] = pydt.Field(
        default=224,
        ge=32,
        le=1024,
        description="Size to resize the image to. If an integer is provided, "
        "the image will be resized to a square of that size. If a "
        "tuple is provided, the image will be resized to the "
        "specified width and height.",
    )

    interpolation: t.Literal["nearest", "nearest-exact", "bilinear", "bicubic"] = (pydt.Field(
        default="bilinear",
        description="Interpolation method to use when resizing the image. "
        "Options are 'nearest', 'nearest-exact', 'bilinear', and "
        "'bicubic'.",
    ))

    max_size: int | None = pydt.Field(
        default=None,
        ge=32,
        le=2048,
        description="The maximum allowed for the longer edge of the resized "
        "image. If the longer edge of the image is greater than "
        "max_size after being resized according to size, "
        "size will be overruled so that the longer edge is equal "
        "to max_size. As a result, the smaller edge may be "
        "shorter than size. This is only supported if size is an "
        "int (or a sequence of length 1 in torchscript mode).",
    )

    antialias: bool = pydt.Field(
        default=True,
        description="Whether to apply an anti-aliasing filter when downsampling "
        "an image. This only has an effect when the image is being "
        "downsampled, and when the interpolation method is "
        "'bilinear' or 'bicubic'.",
    )
