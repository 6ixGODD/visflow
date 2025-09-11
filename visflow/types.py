from __future__ import annotations

import os
import typing as t

import torch
import typing_extensions as te

CriterionFunc = t.Callable[[t.Any, t.Any], torch.Tensor]

PixelValue: t.TypeAlias = (int |
                           t.Tuple[int, int, int] |
                           float |
                           t.Tuple[float, float, float])


def pixel_float(value: PixelValue, /) -> t.Tuple[float, float, float]:
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


def pixel_int(value: PixelValue, /) -> t.Tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    elif isinstance(value, float):
        v = int(value * 255.0)
        return v, v, v
    elif isinstance(value, tuple) and len(value) == 3:
        if all(isinstance(v, int) for v in value):
            return t.cast(t.Tuple[int, int, int], value)
        elif all(isinstance(v, float) for v in value):
            return (int(value[0] * 255.0),
                    int(value[1] * 255.0),
                    int(value[2] * 255.0))
    raise ValueError("Invalid pixel value type")


PathLikes = os.PathLike[str] | str


class Checkpoint(te.TypedDict, total=False):
    epoch: te.Required[int]
    """Epoch number when the checkpoint was saved."""

    model_state_dict: te.Required[t.Dict[str, t.Any]]
    """State dictionary of the model."""

    optimizer_state_dict: t.Dict[str, t.Any] | None
    """State dictionary of the optimizer."""

    scheduler_state_dict: t.Dict[str, t.Any] | None
    """State dictionary of the learning rate scheduler."""

    accuracy: float
    """Accuracy at the time of saving the checkpoint."""

    config: t.Dict[str, t.Any]
    """Configuration dictionary used for training."""

    classes: t.List[str]
    """List of class names."""

    class_to_idx: t.Dict[str, int]
    """Mapping from class names to indices."""

    extra_state: t.Dict[str, t.Any]
    """Any additional state information."""
