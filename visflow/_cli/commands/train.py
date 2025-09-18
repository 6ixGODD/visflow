from __future__ import annotations

import argparse
import ast
import os
import pathlib as p
import typing as t

from visflow._cli.args import BaseArgs
from visflow.pipelines.train import TrainPipeline
from visflow.resources.config import TrainConfig
from visflow.utils import spinner

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    config: str | None = None
    verbose: bool = False
    overrides: t.List[str] = []

    def run(self) -> None:
        spinner.start("Bootstrapping training pipeline...")
        os.environ["FORCE_COLOR"] = "1"
        if self.verbose:
            os.environ["VF_VERBOSE"] = "1"

        if self.config:
            config_path = p.Path(self.config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            train_config = TrainConfig.from_yaml(config_path)
        else:
            train_config = TrainConfig()

        if self.overrides:
            config_dict = train_config.model_dump()
            self._overrides(config_dict, self.overrides)
            train_config = TrainConfig.model_validate(config_dict, strict=True)

        pipeline = TrainPipeline(train_config)
        spinner.succeed("Training pipeline bootstrapped.")
        pipeline()

    def _overrides(
        self, config_dict: t.Dict[str, t.Any], overrides: t.List[str]
    ) -> None:
        i = 0
        while i < len(overrides):
            key = overrides[i]

            if i + 1 >= len(overrides):
                raise ValueError(f"Missing value for key: {key}")

            value_str = overrides[i + 1]
            i += 2

            keys = key.split(".")
            current = config_dict

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            final_key = keys[-1]
            parsed_value = self._parse_value(value_str)
            current[final_key] = parsed_value

    @staticmethod
    def _parse_value(value_str: str, /) -> t.Any:
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        return value_str

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            default=None,
            help="Path to the training configuration file (YAML format). ("
            "default: %(default)s)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output. (default: %(default)s)",
        )
        parser.add_argument(
            "overrides",
            nargs="*",
            help="Configuration overrides in format: key value key value ...",
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        "train",
        help="Train a model using the specified configuration file.",
    )
    Args.add_args(parser)
    parser.set_defaults(func=Args.func)
