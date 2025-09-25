from __future__ import annotations

import abc
import argparse
import typing as t

import pydantic as pydt


class BaseArgs(pydt.BaseModel, abc.ABC):

    @classmethod
    def func(cls, args: argparse.Namespace) -> None:
        instance = cls.from_args(args)
        instance.run()

    @abc.abstractmethod
    def run(self) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> t.Self:
        return cls.model_validate(vars(args), strict=True)
