from __future__ import annotations

import os
import typing as t

import pydantic as pydt
import pydantic_settings as ps

from visflow.resources.config.augmentation import AugmentationConfig
from visflow.resources.config.data import DataConfig
from visflow.resources.config.logging import LoggingConfig
from visflow.resources.config.model import ModelConfig
from visflow.resources.config.normalization import NormalizationConfig
from visflow.resources.config.output import OutputConfig
from visflow.resources.config.resize import ResizeConfig
from visflow.resources.config.testing import TestingConfig
from visflow.resources.config.training import TrainingConfig
from visflow.types import PathLikes


class BaseConfig(ps.BaseSettings):
    model_config: t.ClassVar[ps.SettingsConfigDict] = ps.SettingsConfigDict(
        validate_default=False, extra="allow"
    )

    @classmethod
    def from_yaml(cls, fpath: PathLikes) -> t.Self:
        try:
            import yaml

            with open(fpath, "r") as f:
                content = yaml.safe_load(f)
            return cls.model_validate(content, strict=True)
        except ImportError:
            raise ImportError(
                "`yaml` module is required to load configuration from YAML "
                "files. Please install it using `pip install pyyaml`."
            )

    @classmethod
    def from_json(cls, fpath: PathLikes) -> t.Self:
        import json

        with open(fpath, "r") as f:
            content = json.load(f)
        return cls.model_validate(content, strict=True)

    logging: LoggingConfig = LoggingConfig()
    seed: int = pydt.Field(default=42, description="Random seed for reproducibility")

    def to_file(self, fpath: PathLikes) -> None:
        fpath = os.fspath(fpath)
        ext = os.path.splitext(fpath)[1].lower()
        if ext in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]

                with open(fpath, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self.model_dump(), f)
            except ImportError:
                raise ImportError(
                    "`yaml` module is required to save configuration to YAML "
                    "files. Please install it using `pip install pyyaml`."
                )
        elif ext == ".json":
            import json

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=4)
        else:
            raise ValueError(
                "Unsupported file extension. Use '.yaml', '.yml', or '.json'."
            )

    def to_dict(self) -> t.Dict[str, t.Any]:
        return self.model_dump(mode="json", exclude_none=True)


class TrainConfig(BaseConfig):
    model: ModelConfig = pydt.Field(
        default_factory=ModelConfig, description="Model architecture configuration."
    )

    training: TrainingConfig = pydt.Field(
        default_factory=TrainingConfig,
        description="Training hyperparameters configuration.",
    )

    testing: TestingConfig = pydt.Field(
        default_factory=TestingConfig, description="Testing configuration."
    )

    data: DataConfig = pydt.Field(
        default_factory=DataConfig,
        description="Data loading and preprocessing configuration.",
    )

    resize: ResizeConfig = pydt.Field(
        default_factory=ResizeConfig, description="Image resizing configuration."
    )

    normalization: NormalizationConfig = pydt.Field(
        default_factory=NormalizationConfig,
        description="Image normalization configuration.",
    )

    augmentation: AugmentationConfig = pydt.Field(
        default_factory=AugmentationConfig,
        description="Data augmentation configuration.",
    )

    output: OutputConfig = pydt.Field(
        default_factory=OutputConfig, description="Output and logging configuration."
    )
