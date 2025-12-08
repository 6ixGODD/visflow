from __future__ import annotations

import os
import pathlib
import typing as t

from prototypex.helper.settings import BaseModel
from prototypex.helper.settings.fields import Field
from prototypex.infras.template.jinja2 import Jinja2TemplateFile
from prototypex.infras.template.jinja2 import Jinja2TemplateString
from prototypex.infras.template.jinja2 import WatchedJinja2Template
from prototypex.utils import UNSET
from prototypex.utils import Unset
from pydantic import model_validator

if t.TYPE_CHECKING:
    from prototypex.infras.template import Template
    from prototypex.logger import Logger


class LocalTemplate(BaseModel):
    path: os.PathLike[str] | None = Field(
        default=None,
        description="Path to the local template directory.",
    )

    watch: bool = Field(
        default=False,
        description="Whether to watch the template directory for changes.",
    )

    content: str | None = Field(
        default=None,
        description="Direct content of the template.",
    )

    @model_validator(mode="after")
    def either_path_or_content(self) -> LocalTemplate:
        if not self.path and not self.content:
            raise ValueError("Either 'path' or 'content' must be provided.")
        return self

    def to_template(self, logger: Logger | None = None) -> Template:
        if self.path and self.watch:
            if logger is None:
                raise ValueError("Logger must be provided when 'watch' is enabled.")
            return WatchedJinja2Template(fpath=self.path, logger=logger)
        if self.path:
            return Jinja2TemplateFile(fpath=self.path)
        return Jinja2TemplateString(content=self.content)


class TemplateNormMixin:
    TEMPLATE_SUFFIX: t.ClassVar[str] = "_template"

    @model_validator(mode="before")
    @classmethod
    def normalize_templates(cls, data: dict[str, t.Any]) -> dict[str, t.Any]:
        """Recursively walk `data` and convert any *_template field to
        LocalTemplate."""

        def norm_template_value(v: t.Any) -> t.Any:
            if v is UNSET or isinstance(v, Unset):
                return v
            if isinstance(v, LocalTemplate):
                return v
            if isinstance(v, (str, os.PathLike)):
                return LocalTemplate(path=pathlib.Path(v))
            if isinstance(v, dict):
                return LocalTemplate(**v)
            if isinstance(v, list):
                return [norm_template_value(item) for item in v]
            return v

        def normalize_children(v: t.Any) -> t.Any:
            if isinstance(v, dict):
                out_d: dict[str, t.Any] = {}
                for k, val in v.items():
                    if k.endswith(cls.TEMPLATE_SUFFIX):
                        out_d[k] = norm_template_value(val)
                    else:
                        out_d[k] = normalize_children(val)
                return out_d
            if isinstance(v, list):
                return [normalize_children(i) for i in v]
            return v

        return normalize_children(data)
