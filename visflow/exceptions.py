from __future__ import annotations

import typing as t


class VisflowError(Exception):
    exit_code: t.ClassVar[int] = 1
