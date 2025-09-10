from __future__ import annotations

import typing as t

import pydantic as pydt


class LoggingConfig(pydt.BaseModel):
    backend: t.Literal['native', 'loguru'] = pydt.Field(
        default='native',
        description="Logging backend to use. 'native' uses the standard "
                    "library logging module, 'loguru' uses the Loguru library."
    )

    extra_context: t.Dict[str, t.Any] = pydt.Field(
        default_factory=dict,
        description="Extra context to include in log records"
    )

    loglevel: t.Literal[
        'debug',
        'info',
        'warning',
        'error',
        'critical'
    ] = pydt.Field(
        default='info',
        description="Global log level"
    )
