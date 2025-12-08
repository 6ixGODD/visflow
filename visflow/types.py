from __future__ import annotations

import os
import typing as t

import torch
import typing_extensions as te

CriterionFunc = t.Callable[[t.Any, t.Any], torch.Tensor]

PixelValue: t.TypeAlias = int | tuple[int, int, int] | float | tuple[float, float, float]


def pixel_float(value: PixelValue, /) -> tuple[float, float, float]:
    if isinstance(value, int):
        return value / 255.0, value / 255.0, value / 255.0
    if isinstance(value, float):
        return value, value, value
    if isinstance(value, tuple) and len(value) == 3:
        if all(isinstance(v, int) for v in value):
            return value[0] / 255.0, value[1] / 255.0, value[2] / 255.0
        if all(isinstance(v, float) for v in value):
            return value
    raise ValueError("Invalid pixel value type")


def pixel_int(value: PixelValue, /) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if isinstance(value, float):
        v = int(value * 255.0)
        return v, v, v
    if isinstance(value, tuple) and len(value) == 3:
        if all(isinstance(v, int) for v in value):
            return t.cast(tuple[int, int, int], value)
        if all(isinstance(v, float) for v in value):
            return int(value[0] * 255.0), int(value[1] * 255.0), int(value[2] * 255.0)
    raise ValueError("Invalid pixel value type")


PathLikes = os.PathLike[str] | str


class Checkpoint(te.TypedDict, total=False):
    epoch: te.Required[int]
    """Epoch number when the checkpoint was saved."""

    model_state_dict: te.Required[dict[str, t.Any]]
    """State dictionary of the model."""

    optimizer_state_dict: dict[str, t.Any] | None
    """State dictionary of the optimizer."""

    scheduler_state_dict: dict[str, t.Any] | None
    """State dictionary of the learning rate scheduler."""

    accuracy: float
    """Accuracy at the time of saving the checkpoint."""

    config: dict[str, t.Any]
    """Configuration dictionary used for training."""

    classes: list[str]
    """List of class names."""

    class_to_idx: dict[str, int]
    """Mapping from class names to indices."""

    extra_state: dict[str, t.Any]
    """Any additional state information."""
