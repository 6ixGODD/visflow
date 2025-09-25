from __future__ import annotations

import sys
import typing as t

from visflow.resources.logger import BaseLogger, LoggerBackend
from visflow.resources.logger.types import LoggingTarget, LogLevel
from visflow.utils import singleton

try:
    from loguru import logger as _logger
except ImportError:
    raise

_LevelMapper: t.Dict[LogLevel, str] = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
    "critical": "CRITICAL",
}


@singleton
class LoguruBackend(LoggerBackend):

    def __init__(self) -> None:
        self._loguru = _logger
        self._loguru.remove()
        self._handler_ids: t.List[int] = []
        self._is_setup = False

    def setup_handlers(self, targets: t.List[LoggingTarget]) -> None:
        if self._is_setup:
            return

        for target in targets:
            level = target.loglevel
            if target.logname == "stdout":
                handler_id = self._loguru.add(
                    sys.stdout,
                    level=_LevelMapper.get(level, "INFO"),
                    colorize=True,
                    serialize=False,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )
            elif target.logname == "stderr":
                handler_id = self._loguru.add(
                    sys.stderr,
                    level=_LevelMapper.get(level, "ERROR"),
                    colorize=True,
                    serialize=False,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )
            else:
                handler_id = self._loguru.add(
                    target.logname,
                    level=_LevelMapper.get(level, "INFO"),
                    colorize=False,
                    serialize=True,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )

            self._handler_ids.append(handler_id)

        self._is_setup = True

    def log(self, msg: str, /, level: LogLevel, **context: t.Any) -> None:
        self._loguru.bind(**context).log(_LevelMapper.get(level, "INFO"), msg)

    def sync(self) -> None:
        pass  # do nothing, loguru is synchronous

    def close(self) -> None:
        for handler_id in self._handler_ids:
            try:
                self._loguru.remove(handler_id)
            except ValueError:
                pass  # Handler already removed
        self._handler_ids.clear()
        self._is_setup = False


class LoguruLogger(BaseLogger):

    def __init__(self,
                 targets: t.Sequence[LoggingTarget] | None = None,
                 initial_ctx: t.Dict[str, t.Any] | None = None):
        backend = LoguruBackend()
        super().__init__(backend, targets, initial_ctx)
