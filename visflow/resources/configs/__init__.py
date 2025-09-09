from __future__ import annotations

import os
import typing as t

import pydantic as pydt
import pydantic_settings as ps

from visflow.resources.configs.augmentation import AugmentationConfig
from visflow.resources.configs.data import DataConfig
from visflow.resources.configs.export import ExportConfig
from visflow.resources.configs.logging import LoggingConfig
from visflow.resources.configs.model import ModelConfig
from visflow.resources.configs.normalization import NormalizationConfig
from visflow.resources.configs.output import OutputConfig
from visflow.resources.configs.resize import ResizeConfig
from visflow.resources.configs.testing import TestingConfig
from visflow.resources.configs.training import TrainingConfig


class BaseConfig(ps.BaseSettings):
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

    def to_file(self, fpath: t.AnyStr | os.PathLike[t.AnyStr]) -> None:
        fpath = os.fspath(fpath)
        ext = os.path.splitext(fpath)[1].lower()
        if ext in {'.yaml', '.yml'}:
            try:
                import yaml
                with open(fpath, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self.model_dump(), f)
            except ImportError:
                raise ImportError(
                    '`yaml` module is required to save configuration to YAML '
                    'files. Please install it using `pip install pyyaml`.'
                )
        elif ext == '.json':
            import json
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(self.model_dump(), f, indent=4)  # type: ignore
        else:
            raise ValueError(
                "Unsupported file extension. Use '.yaml', '.yml', or '.json'."
            )


class TrainConfig(BaseConfig):
    model: ModelConfig = pydt.Field(
        default_factory=ModelConfig,
        description="Model architecture configuration."
    )

    training: TrainingConfig = pydt.Field(
        default_factory=TrainingConfig,
        description="Training hyperparameters configuration."
    )

    testing: TestingConfig = pydt.Field(
        default_factory=TestingConfig,
        description="Testing configuration."
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


class TestConfig(BaseConfig):
    pass
