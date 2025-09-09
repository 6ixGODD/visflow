from __future__ import annotations

import typing as t


class Pipeline(t.Protocol):
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        pass
