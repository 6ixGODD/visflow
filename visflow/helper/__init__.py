from __future__ import annotations

import inspect
import typing as t

import loguru


class LoggingMixin:
    __logtag__: t.ClassVar[str]

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and not getattr(cls, "__logtag__", None):
            cls.__logtag__ = cls.__name__

    def __init__(self) -> None:
        self.logger = loguru.logger.bind(tag=self.__logtag__)
