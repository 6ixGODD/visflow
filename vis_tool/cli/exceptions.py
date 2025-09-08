from __future__ import annotations

import typing as t


class CLIException(Exception):
    exit_code: t.ClassVar[int] = -1


class InvalidArgumentException(CLIException):
    exit_code = 3

    def __init__(self, arguments: str | t.List[str]) -> None:
        if isinstance(arguments, str):
            arguments = [arguments]
        super().__init__(f"Invalid argument(s): {', '.join(arguments)}")
        self.arguments = arguments
