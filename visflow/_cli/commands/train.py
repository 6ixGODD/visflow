from __future__ import annotations

import argparse

from visflow._cli.helpers.args import BaseArgs


class Args(BaseArgs):
    def _func(self) -> None:
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        pass
