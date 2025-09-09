from __future__ import annotations

import os
import typing as t

import pydantic as pydt
import pydantic_settings as ps

from visflow.configs.augmentation import AugmentationConfig
from visflow.configs.data import DataConfig
from visflow.configs.export import ExportConfig
from visflow.configs.logging import LoggingConfig
from visflow.configs.model import ModelConfig
from visflow.configs.normalization import NormalizationConfig
from visflow.configs.output import OutputConfig
from visflow.configs.resize import ResizeConfig
from visflow.configs.training import TrainingConfig


class Config(ps.BaseSettings):
    model_config: t.ClassVar[pydt.ConfigDict] = ps.SettingsConfigDict(
        validate_default=False,
        extra='allow'
    )

    @classmethod
    def from_yaml(cls, fpath: t.AnyStr | os.PathLike[t.AnyStr]) -> t.Self:
        try:
            import yaml
            with open(fpath, 'r') as f:
                content = yaml.safe_load(f)
            return cls.model_validate(content, strict=True)
        except ImportError:
            raise ImportError(
                '`yaml` module is required to load configuration from YAML '
                'files. Please install it using `pip install pyyaml`.'
            )

    @classmethod
    def from_json(cls, fpath: t.AnyStr | os.PathLike[t.AnyStr]) -> t.Self:
        import json
        with open(fpath, 'r') as f:
            content = json.load(f)
        return cls.model_validate(content, strict=True)

    logging: LoggingConfig = LoggingConfig()
    seed: int = 42


class TrainConfig(Config):
    model: ModelConfig = pydt.Field(
        default_factory=ModelConfig,
        description="Model architecture configuration."
    )

    training: TrainingConfig = pydt.Field(
        default_factory=TrainingConfig,
        description="Training hyperparameters configuration."
    )

    data: DataConfig = pydt.Field(
        default_factory=DataConfig,
        description="Data loading and preprocessing configuration."
    )

    resize: ResizeConfig = pydt.Field(
        default_factory=ResizeConfig,
        description="Image resizing configuration."
    )

    normalization: NormalizationConfig = pydt.Field(
        default_factory=NormalizationConfig,
        description="Image normalization configuration."
    )

    augmentation: AugmentationConfig = pydt.Field(
        default_factory=AugmentationConfig,
        description="Data augmentation configuration."
    )

    export: ExportConfig = pydt.Field(
        default_factory=ExportConfig,
        description="Model export configuration."
    )

    output: OutputConfig = pydt.Field(
        default_factory=OutputConfig,
        description="Output and logging configuration."
    )


class TestConfig(Config):
    pass


class PreprocessConfig(Config):
    pass
