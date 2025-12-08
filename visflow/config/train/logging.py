from __future__ import annotations

import os
import pathlib
import sys
import typing as t

from pydantic import model_validator

from visflow.helper.settings import BaseModel
from visflow.helper.settings.fields import Field


class LoggingTarget(BaseModel):
    logname: t.Literal["stdout", "stderr"] | os.PathLike[str] = Field(
        default="stdout",
        description="Name of the target, can be 'stdout', 'stderr', or a file path",
        examples=["stdout", "stderr", pathlib.Path("./logs/train.log")],
    )

    loglevel: t.Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Log level for this target",
    )


class LoggingConfig(BaseModel):
    targets: list[LoggingTarget] = Field(
        default_factory=lambda: [LoggingTarget(logname="stdout", loglevel="debug")],
        description="List of logging targets",
    )

    def _setup_logger(self) -> None:
        from loguru import logger

        # Clear existing handlers
        logger.remove()

        # Set up each logging target
        for target in self.targets:
            level = target.loglevel.upper()
            if target.logname == "stdout":
                logger.add(sys.stdout, level=level)
                continue
            if target.logname == "stderr":
                logger.add(sys.stderr, level=level)
                continue
            sink = target.logname
            logger.add(sink, level=level)

    @model_validator(mode="after")
    def setup_logger(self) -> LoggingConfig:
        self._setup_logger()
        return self
