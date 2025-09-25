from __future__ import annotations

import argparse
import typing as t

from visflow._cli.args import BaseArgs
from visflow.pipelines.init import InitPipeline

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    proxy: str | None = None

    def run(self) -> None:
        pipeline = InitPipeline(proxy=self.proxy)
        pipeline()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proxy",
                            "-p",
                            type=str,
                            default=None,
                            help="Proxy server to use for downloading files (e.g., "
                            "http://proxyserver:port)")


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser("init", help="Initialize the VisFlow environment")
    Args.add_args(parser)
    parser.set_defaults(func=Args.func)
