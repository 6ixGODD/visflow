from __future__ import annotations

import typing as t

import pydantic as pydt


class PlateauConfig(pydt.BaseModel):
    mode: t.Literal["min", "max"] = pydt.Field(
        default="min",
        description="One of min, max. In min mode, lr will be reduced when "
        "the quantity monitored has stopped decreasing; in max "
        "mode it will be reduced when the quantity monitored has "
        "stopped increasing. Default: 'min'.",
    )
    factor: float = pydt.Field(
        default=0.1,
        description="Factor by which the learning rate will be reduced. new_lr "
        "= lr * factor. Default: 0.1.",
        gt=0.0,
        lt=1.0,
    )
    patience: int = pydt.Field(
        default=10,
        description="Number of epochs with no improvement after which learning "
        "rate will be reduced. For example, if patience = 2, then "
        "we will wait for 2 epochs to see if the quantity "
        "monitored improves, and if it doesn't improve for 2 "
        "consecutive epochs, we will reduce the learning rate. "
        "Default: 10.",
        ge=0,
    )
    threshold: float = pydt.Field(
        default=1e-4,
        description="Threshold for measuring the new optimum, to only focus "
        "on significant changes. Default: 1e-4.",
        ge=0.0,
    )
    threshold_mode: t.Literal["rel", "abs"] = pydt.Field(
        default="rel",
        description="One of rel, abs. In rel mode, dynamic_threshold = "
        "best * (1 + threshold) in 'max' mode or best * (1 - "
        "threshold) in 'min' mode. In abs mode, dynamic_threshold "
        "= best + threshold in 'max' mode or best - threshold in "
        "'min' mode. Default: 'rel'.",
    )
    cooldown: int = pydt.Field(
        default=0,
        description="Number of epochs to wait before resuming normal "
        "operation after lr has been reduced. Default: 0.",
        ge=0,
    )
    min_lr: float | list[float] = pydt.Field(
        default=0,
        description="A scalar or a list of scalars. A lower bound on the "
        "learning rate of all param groups or each group "
        "respectively. Default: 0.",
        ge=0.0,
    )
    eps: float = pydt.Field(
        default=1e-8,
        description="Minimal decay applied to lr. If the difference between "
        "new and old lr is smaller than eps, the update is "
        "ignored. Default: 1e-8.",
        ge=0.0,
    )
