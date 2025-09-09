from __future__ import annotations

import typing as t

import pydantic as pydt


class LoggingTarget(pydt.BaseModel):
    logname: t.Literal['stdout', 'stderr'] | str = pydt.Field(
        default='stdout',
        description="Name of the target, can be 'stdout', 'stderr', or a file "
                    "path"
    )

    loglevel: t.Literal[
        'debug', 'info', 'warning', 'error', 'critical'
    ] = pydt.Field(
        default='info',
        description="Log level for this target"
    )


class LoggingConfig(pydt.BaseModel):
    backend: t.Literal['native', 'loguru'] = 'native'

    targets: t.List[LoggingTarget] = [
        LoggingTarget(logname='stdout', loglevel='debug'),
        LoggingTarget(logname='stderr', loglevel='error'),
    ]

    extra_context: t.Dict[str, t.Any] = {}
