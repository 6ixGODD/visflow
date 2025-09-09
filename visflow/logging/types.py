from __future__ import annotations

import typing as t

from visflow.types import FileLikes

LogLevel: t.TypeAlias = t.Literal[
    'debug',
    'info',
    'warning',
    'error',
    'critical'
]


class LoggingTarget:
    __slots__: t.Collection[str] = ('logname', 'loglevel')

    def __init__(
        self,
        *,
        logname: t.Literal['stdout', 'stderr'] | FileLikes = 'stdout',
        loglevel: LogLevel = 'info'
    ) -> None:
        self.logname = logname
        self.loglevel = loglevel
