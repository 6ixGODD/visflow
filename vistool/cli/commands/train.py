from __future__ import annotations

import argparse

from vistool.cli.helpers.args import BaseArgs


class Args(BaseArgs):
    __slots__ = ()

    def _func(self) -> None:
        pass

    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        pass
