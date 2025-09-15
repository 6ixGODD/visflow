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
    overrides: list[str] = []  # 新增字段

    def run(self) -> None:
        print("🔧 Loading configuration...")

        # 加载基础配置
        if self.config:
            config_path = p.Path(self.config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            train_config = TrainConfig.from_yaml(config_path)
            print(f"✅ Loaded config from: {config_path}")
        else:
            train_config = TrainConfig()
            print("✅ Using default configuration")

        # 打印原始配置
        print("\n📋 Original Configuration:")
        print(json.dumps(train_config.to_dict(), indent=2))

        # 应用命令行覆盖
        if self.overrides:
            print(f"\n🎯 Applying {len(self.overrides) // 2} overrides...")
            config_dict = train_config.model_dump()
            self._apply_overrides(config_dict, self.overrides)
            train_config = TrainConfig.model_validate(config_dict, strict=True)
            print("✅ Overrides applied successfully!")

            # 打印最终配置
            print("\n📋 Final Configuration (after overrides):")
            print(json.dumps(train_config.to_dict(), indent=2))
        else:
            print("\n💡 No overrides provided")

        # 暂时不运行 pipeline，只显示配置
        print(
            "\n🚀 Configuration ready! (Pipeline execution disabled for testing)"
            )
        print("=" * 60)

        # 可选：显示一些关键配置项
        print("Key Configuration Summary:")
        print(
            f"  Model Architecture: {getattr(train_config.model, 'architecture', 'N/A')}"
            )
        print(
            f"  Model Pretrained: {getattr(train_config.model, 'pretrained', 'N/A')}"
            )
        print(f"  Resize Size: {getattr(train_config.resize, 'size', 'N/A')}")
        print(f"  Random Seed: {train_config.seed}")
        # spinner.start('Bootstrapping training pipeline...')
        # os.environ['FORCE_COLOR'] = '1'
        # if self.verbose:
        #     os.environ['VF_VERBOSE'] = '1'
        #
        # if self.config:
        #     config_path = p.Path(self.config)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"Config file not found: {config_path}")
        #     train_config = TrainConfig.from_yaml(config_path)
        # else:
        #     train_config = TrainConfig()
        #
        # if self.overrides:
        #     config_dict = train_config.model_dump()
        #     self._overrides(config_dict, self.overrides)
        #     train_config = TrainConfig.model_validate(config_dict, strict=True)
        #
        # pipeline = TrainPipeline(train_config)
        # spinner.succeed('Training pipeline bootstrapped.')
        # pipeline()

    def _overrides(
        self,
        config_dict: t.Dict[str, t.Any],
        overrides: t.List[str]
    ) -> None:
        i = 0
        while i < len(overrides):
            key = overrides[i]

            if i + 1 >= len(overrides):
                raise ValueError(f"Missing value for key: {key}")

            value_str = overrides[i + 1]
            i += 2

            keys = key.split('.')
            current = config_dict

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            final_key = keys[-1]
            parsed_value = self._parse_value(value_str)
            current[final_key] = parsed_value

    @staticmethod
    def _parse_value(value_str: str) -> t.Any:
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        return value_str

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--config', '-c',
            type=str,
            default=None,
            help='Path to the training configuration file (YAML format). ('
                 'default: %(default)s)'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output. (default: %(default)s)'
        )
        parser.add_argument(
            'overrides',
            nargs='*',
            help='Configuration overrides in format: key value key value ...'
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        'train',
        help='Train a model using the specified configuration file.',
    )
    Args.add_args(parser)
    parser.set_defaults(func=Args.func)
