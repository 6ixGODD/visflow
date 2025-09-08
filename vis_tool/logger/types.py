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
LoggingTargetName: t.TypeAlias = t.Union[t.Literal['stdout', 'stderr'], str]


class SizeBasedRotation(pydt.BaseModel):
    max_size: int
    backup_count: int


class TimeBasedRotation(pydt.BaseModel):
    interval: int
    backup_count: int


class Rotation(pydt.BaseModel):
    size_based: SizeBasedRotation | None
    time_based: TimeBasedRotation | None


class LoggingTarget(pydt.BaseModel):
    logname: t.Literal['stdout', 'stderr'] | str

    loglevel: LogLevel

    rotation: Rotation | None
