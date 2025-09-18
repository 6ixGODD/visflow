from __future__ import annotations

import abc


class BasePipeline(abc.ABC):
    __slots__ = ("_completed",)
    _completed: bool

    @abc.abstractmethod
    def __call__(self) -> None: ...

    @property
    def completed(self) -> bool:
        return self._completed
