from __future__ import annotations

import abc
import types
import typing as t


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
