from __future__ import annotations

import abc
import argparse
import typing as t

import visflow.helpers.slotted as slotted


class BaseArgs(slotted.SlottedDataClass):
    __slots__ = ('config',)

    config: str

    @classmethod
    def func(cls, args: argparse.Namespace) -> None:
        instance = cls.from_args(args)
        instance._func()

    @abc.abstractmethod
    def _func(self) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> t.Self:
        return cls.from_dict(vars(args))
