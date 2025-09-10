from __future__ import annotations

import argparse
import os
import pathlib as p
import typing as t

from visflow.pipelines.train import TrainPipeline
from visflow.resources.configs import TrainConfig

T = t.TypeVar('T')

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction

from visflow._cli.args import BaseArgs


class Args(BaseArgs):
    __slots__ = ('config', 'verbose')

    _field_defaults = {'verbose': False}

    config: str
    verbose: bool

    def _func(self) -> None:
        if self.verbose:
            os.environ['VF_VERBOSE'] = '1'
        config_path = p.Path(self.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        train_config = TrainConfig.from_yaml(config_path)
        pipeline = TrainPipeline(train_config)
        pipeline()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--config', '-c',
            type=str,
            required=True,
            help='Path to the training configuration file (YAML format).'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output.'
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        'train',
        help='Train a model using the specified configuration file.',
    )
    Args.add_args(parser)
    parser.set_defaults(func=Args.func)
