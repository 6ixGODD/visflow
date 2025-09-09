from __future__ import annotations

import typing as t

import torch

CriterionFunc = t.Callable[[t.Any, t.Any], torch.Tensor]
PixelValue: t.TypeAlias = (int |
                           t.Tuple[int, int, int] |
                           float |
                           t.Tuple[float, float, float])


def pixel_float(value: PixelValue) -> t.Tuple[float, float, float]:
    if isinstance(value, int):
        return value / 255.0, value / 255.0, value / 255.0
    elif isinstance(value, float):
        return value, value, value
    elif isinstance(value, tuple) and len(value) == 3:
        if all(isinstance(v, int) for v in value):
            return value[0] / 255.0, value[1] / 255.0, value[2] / 255.0
        elif all(isinstance(v, float) for v in value):
            return value
    raise ValueError("Invalid pixel value type")


def pixel_int(value: PixelValue) -> t.Tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    elif isinstance(value, float):
        v = int(value * 255.0)
        return v, v, v
    elif isinstance(value, tuple) and len(value) == 3:
        if all(isinstance(v, int) for v in value):
            return value
        elif all(isinstance(v, float) for v in value):
            return (int(value[0] * 255.0),
                    int(value[1] * 255.0),
                    int(value[2] * 255.0))
    raise ValueError("Invalid pixel value type")
