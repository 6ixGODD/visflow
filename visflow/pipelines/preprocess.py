from __future__ import annotations

import typing as t


class PreprocessPipeline:
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
