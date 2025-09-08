from __future__ import annotations

import datetime
import sys
import typing as t

import vis_tool.utils as utils
from vis_tool.logger import Logger, LoggerBackend
from vis_tool.logger.types import LoggingTarget, LogLevel, Rotation

try:
    from loguru import logger as _logger
except ImportError:
    raise

_LEVEL_MAP: t.Dict[LogLevel, str] = {
    'debug': 'DEBUG',
    'info': 'INFO',
    'warning': 'WARNING',
    'error': 'ERROR',
    'critical': 'CRITICAL',
}


def _get_loguru_rotation(cfg: Rotation | None) -> t.Tuple[
    t.Optional[datetime.timedelta | str],
    t.Optional[int],
]:
    if cfg is None:
        return None, None

    if cfg.size_based:
        return f'{cfg.size_based.max_size} MB', cfg.size_based.backup_count
    elif cfg.time_based:
        return (
            datetime.timedelta(hours=cfg.time_based.interval),
            cfg.time_based.backup_count,
        )
    else:
        return None, None


@utils.singleton
class LoguruBackend(LoggerBackend):

    def __init__(self):
        self._loguru = _logger
        self._loguru.remove()
        self._handler_ids: t.List[int] = []
        self._is_setup = False

    def setup_handlers(self, targets: t.List[LoggingTarget]) -> None:
        if self._is_setup:
            return

        for target in targets:
            rotation, retention = _get_loguru_rotation(target.rotation)
            level = target.loglevel

            if target.logname == 'stdout':
                handler_id = self._loguru.add(
                    sys.stdout,
                    level=_LEVEL_MAP.get(level, 'INFO'),
                    colorize=True,
                    serialize=False,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )
            elif target.logname == 'stderr':
                handler_id = self._loguru.add(
                    sys.stderr,
                    level=_LEVEL_MAP.get(level, 'ERROR'),
                    colorize=True,
                    serialize=False,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )
            else:
                handler_id = self._loguru.add(
                    target.logname,
                    level=_LEVEL_MAP.get(level, 'INFO'),
                    rotation=rotation,
                    retention=retention,
                    colorize=False,
                    serialize=True,
                    backtrace=False,
                    diagnose=False,
                    catch=False,
                )

            self._handler_ids.append(handler_id)

        self._is_setup = True

    def log(
        self,
        msg: str, /,
        level: LogLevel,
        **context: t.Any
    ) -> None:
        self._loguru.bind(**context).lg(
            _LEVEL_MAP.get(level, 'INFO'),
            msg,
        )

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


class LoguruLogger(Logger):
    def __init__(
        self,
        targets: t.Sequence[LoggingTarget] | None = None,
        initial_ctx: t.Dict[str, t.Any] | None = None,
    ):
        backend = LoguruBackend()
        super().__init__(backend, targets, initial_ctx)
