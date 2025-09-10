from __future__ import annotations

import abc


class BasePipeline(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> None: ...
