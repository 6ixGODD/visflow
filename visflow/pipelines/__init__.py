from __future__ import annotations

import abc
import typing as t


class BasePipeline(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
