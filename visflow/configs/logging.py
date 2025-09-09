from __future__ import annotations

import typing as t

import pydantic as pydt


class LoggingConfig(pydt.BaseModel):
    backend: t.Literal['native', 'loguru'] = 'native'
    extra_context: t.Dict[str, t.Any] = {}
