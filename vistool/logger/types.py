from __future__ import annotations

import typing as t

import pydantic as pydt

LogLevel: t.TypeAlias = t.Literal[
    'debug',
    'info',
    'warning',
    'error',
    'critical'
]


class LoggingTarget(pydt.BaseModel):
    logname: t.Literal['stdout', 'stderr'] | str
    loglevel: LogLevel
