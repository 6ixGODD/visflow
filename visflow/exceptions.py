from __future__ import annotations

import typing as t


class VisflowError(Exception):
    exit_code: t.ClassVar[int] = 1

    def __init__(self, message: str, /):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.message!r})'


class InvalidConfigurationError(VisflowError):
    exit_code: t.ClassVar[int] = 2

    def __init__(
        self,
        message: str = 'Invalid configuration',
        /,
        params: dict[str, t.Any] | None = None
    ):
        super().__init__(message)
        self.params = params or {}

    def __str__(self) -> str:
        if self.params:
            params_str = ', '.join(f'{k}={v!r}' for k, v in self.params.items())
            return f'{self.message} ({params_str})'
        return self.message

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'({self.message!r}, params={self.params!r})')
