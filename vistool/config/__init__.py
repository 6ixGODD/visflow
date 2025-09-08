from __future__ import annotations

import os
import typing as t

import pydantic as pydt
import pydantic_settings as ps

from vistool.config.augmentation import AugmentationConfig
from vistool.config.data import DataConfig
from vistool.config.export import ExportConfig
from vistool.config.model import ModelConfig
from vistool.config.normalization import NormalizationConfig
from vistool.config.output import OutputConfig
from vistool.config.training import TrainingConfig


class TrainConfig(ps.BaseSettings):
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

    model_config: t.ClassVar[pydt.ConfigDict] = ps.SettingsConfigDict(
        env_prefix='VISTOOL__',
        validate_default=False,
        env_nested_delimiter='__',
        env_file='.env',
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


config = TrainConfig()


def set(c: TrainConfig) -> None:
    global config
    config = c


def get() -> TrainConfig:
    return config
