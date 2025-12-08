from __future__ import annotations

import pathlib
import typing as t

import pydantic as pydt
import pydantic_settings as ps

from visflow.config.augmentation import AugmentationConfig
from visflow.config.data import DataConfig
from visflow.config.logging import LoggingConfig
from visflow.config.model import ModelConfig
from visflow.config.normalization import NormalizationConfig
from visflow.config.output import OutputConfig
from visflow.config.resize import ResizeConfig
from visflow.config.testing import TestingConfig
from visflow.config.training import TrainingConfig
from visflow.types import PathLikes


class BaseConfig(ps.BaseSettings):
    model_config: t.ClassVar[ps.SettingsConfigDict] = ps.SettingsConfigDict(
        validate_default=False, extra="allow"
    )

    @classmethod
    def from_yaml(cls, fpath: PathLikes) -> t.Self:
        try:
            import yaml

            with pathlib.Path(fpath).open() as f:
                content = yaml.safe_load(f)
            return cls.model_validate(content, strict=True)
        except ImportError as e:
            raise ImportError(
                "`yaml` module is required to load configuration from YAML "
                "files. Please install it using `pip install pyyaml`."
            ) from e

    @classmethod
    def from_json(cls, fpath: PathLikes) -> t.Self:
        import json

        with pathlib.Path(fpath).open() as f:
            content = json.load(f)
        return cls.model_validate(content, strict=True)

    logging: LoggingConfig = LoggingConfig()
    seed: int = pydt.Field(default=42, description="Random seed for reproducibility")

    def to_file(self, fpath: PathLikes) -> None:
        fpath = pathlib.Path(fpath)
        ext = fpath.suffix.lower()
        if ext in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]

                with pathlib.Path(fpath).open("w", encoding="utf-8") as f:
                    yaml.safe_dump(self.model_dump(), f)
            except ImportError as e:
                raise ImportError(
                    "`yaml` module is required to save configuration to YAML "
                    "files. Please install it using `pip install pyyaml`."
                ) from e
        elif ext == ".json":
            import json

            with fpath.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=4)
        else:
            raise ValueError("Unsupported file extension. Use '.yaml', '.yml', or '.json'.")

    def to_dict(self) -> dict[str, t.Any]:
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


class TestConfig(BaseConfig):
    device: t.Literal["cpu", "cuda"] = pydt.Field(
        "cpu",
        description="Device to run the evaluation on ('cpu' or 'cuda').",
    )

    shuffle: bool = pydt.Field(
        True,
        description="Whether to shuffle the dataset before evaluation.",
    )

    batch_size: int = pydt.Field(
        default=32, ge=1, le=512, description="Number of samples per batch during evaluation."
    )

    checkpoint_files: list[str] = pydt.Field(
        [],
        description="List of checkpoint file paths to be evaluated.",
        min_length=1,
    )

    test_set: str = pydt.Field(
        "test",
        description="Name of the test dataset split to be used.",
    )

    output_dir: str = pydt.Field(
        default="./output",
        description="Directory where training outputs will be saved.",
    )

    experiment_name: str = pydt.Field(
        default="test", min_length=1, description="Unique name for the experiment."
    )
