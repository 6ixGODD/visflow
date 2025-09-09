from __future__ import annotations

import typing as t

from visflow.pipelines import BasePipeline


class TestPipeline(BasePipeline):

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pass
