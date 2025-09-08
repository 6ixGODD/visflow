from __future__ import annotations

import abc
import inspect
import types
import typing as t

from vistool.logger import Logger


class AsyncContextMixin(abc.ABC):
    @abc.abstractmethod
    async def init(self) -> None:
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        pass

    async def __aenter__(self) -> t.Self:
        await self.init()
        return self

    async def __aexit__(
        self,
        exc_type: t.Type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: types.TracebackType | None
    ) -> t.Literal[False]:
        await self.close()
        if exc_type is not None:
            raise exc_val.with_traceback(traceback) from exc_val
        return False


class ContextMixin(abc.ABC):
    @abc.abstractmethod
    def init(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> t.Self:
        self.init()
        return self

    def __exit__(
        self,
        exc_type: t.Type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: types.TracebackType | None
    ) -> t.Literal[False]:
        self.close()
        if exc_type is not None:
            raise exc_val.with_traceback(traceback) from exc_val
        return False


_SendT = t.TypeVar('_SendT')
_RecvT = t.TypeVar('_RecvT', bound=t.AsyncIterable)


class StatelessDuplexMixin(t.Protocol[_SendT, _RecvT]):
    async def send(self, data: _SendT) -> None:
        pass

    def recv(self) -> _RecvT:
        pass


class LoggingTagMixin:
    __logging_tag__: t.ClassVar[str]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not inspect.isabstract(cls) and not hasattr(cls, '__logging_tag__'):
            raise TypeError(
                f"{cls.__name__} must define __logging_tag__ class variable"
            )

    def __init__(self, logger: Logger):
        self.logger = logger.with_tag(self.__logging_tag__)
