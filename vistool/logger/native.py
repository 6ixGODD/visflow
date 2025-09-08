from __future__ import annotations

import json
import logging
import logging.handlers
import pathlib as p
import sys
import typing as t

import vistool.utils as utils
import vistool.utils.ansi as ansi_utils
from vistool.logger import Logger, LoggerBackend
from vistool.logger.types import LoggingTarget, LogLevel

_LEVEL_MAP: t.Dict[LogLevel, int] = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


class ContextFormatter(logging.Formatter):
    """Enhanced formatter with ANSI colors and improved layout."""

    LEVEL_COLORS = {
        logging.DEBUG: (ansi_utils.ANSIFormatter.FG.GRAY,
                        ansi_utils.ANSIFormatter.STYLE.DIM),
        logging.INFO: (ansi_utils.ANSIFormatter.FG.BRIGHT_CYAN,),
        logging.WARNING: (ansi_utils.ANSIFormatter.FG.BRIGHT_YELLOW,
                          ansi_utils.ANSIFormatter.STYLE.BOLD),
        logging.ERROR: (ansi_utils.ANSIFormatter.FG.BRIGHT_RED,
                        ansi_utils.ANSIFormatter.STYLE.BOLD),
        logging.CRITICAL: (ansi_utils.ANSIFormatter.FG.BRIGHT_RED,
                           ansi_utils.ANSIFormatter.BG.WHITE,
                           ansi_utils.ANSIFormatter.STYLE.BOLD),
    }

    COMPONENT_STYLES = {
        'timestamp': (ansi_utils.ANSIFormatter.FG.GRAY,),
        'logger': (ansi_utils.ANSIFormatter.FG.MAGENTA,),
        'tag': (ansi_utils.ANSIFormatter.FG.CYAN,
                ansi_utils.ANSIFormatter.STYLE.BOLD),
        'arrow': (ansi_utils.ANSIFormatter.FG.GRAY,),
        'context': (ansi_utils.ANSIFormatter.FG.GRAY,),
        # Same as arrow for consistency
    }

    def __init__(self, is_console: bool = False, use_colors: bool = True):
        super().__init__(datefmt='%Y-%m-%d %H:%M:%S')
        self.is_console = is_console
        self.use_colors = (use_colors and
                           ansi_utils.ANSIFormatter.supports_color())
        if self.use_colors:
            ansi_utils.ANSIFormatter.enable(True)

    @staticmethod
    def _extract_context(record: logging.LogRecord) -> t.Dict[str, t.Any]:
        """Extract context from log record, excluding standard fields."""
        excluded = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
            'relativeCreated', 'thread', 'threadName', 'processName', 'process',
            'getMessage', 'exc_info', 'exc_text', 'stack_info', 'message'
        }

        context = {}
        for key, value in record.__dict__.items():
            if not key.startswith('_') and key not in excluded:
                try:
                    json.dumps(value)  # Test serializability
                    context[key] = value
                except (TypeError, ValueError):
                    context[key] = str(value)
        return context

    def _format_logger_with_tag(
        self,
        name: str,
        context: t.Dict[str, t.Any]
    ) -> str:
        """Format logger name with optional TAG."""
        tag = context.get('TAG')

        if tag:
            # Calculate available space: 40 total - 1 for dot = 39
            available_space = 39
            name_len = len(name)
            tag_len = len(tag)
            total_needed = name_len + tag_len

            if total_needed <= available_space:
                # Both fit, no truncation needed
                truncated_name = name
                truncated_tag = tag
            else:
                # Need to truncate
                if name_len <= 8:  # Keep short names intact
                    truncated_name = name
                    remaining = available_space - name_len
                    if tag_len > remaining:
                        truncated_tag = f"...{tag[-(remaining - 3):]}"
                    else:
                        truncated_tag = tag
                else:
                    # Truncate name, keep reasonable tag space
                    max_tag_space = min(tag_len, 8)  # At most 8 chars for tag
                    max_name_space = available_space - max_tag_space

                    if name_len > max_name_space:
                        truncated_name = f"...{name[-(max_name_space - 3):]}"
                    else:
                        truncated_name = name
                        max_tag_space = available_space - len(truncated_name)

                    if tag_len > max_tag_space:
                        truncated_tag = f"...{tag[-(max_tag_space - 3):]}"
                    else:
                        truncated_tag = tag

            full_display = f"{truncated_name}.{truncated_tag}"

            if self.use_colors:
                colored_name = ansi_utils.ANSIFormatter.format(
                    truncated_name,
                    *self.COMPONENT_STYLES['logger']
                )
                colored_tag = ansi_utils.ANSIFormatter.format(
                    truncated_tag,
                    *self.COMPONENT_STYLES['tag']
                )
                colored_dot = ansi_utils.ANSIFormatter.format(
                    ".",
                    ansi_utils.ANSIFormatter.FG.GRAY
                )
                display = f"{colored_name}{colored_dot}{colored_tag}"
            else:
                display = full_display

            # Pad to 40 characters
            if len(full_display) < 40:
                display += " " * (40 - len(full_display))
            else:
                # Should not happen with our calculation, but just in case
                display = display[:40] if not self.use_colors else display
        else:
            # Without TAG: use full 40 characters
            if len(name) > 40:
                truncated_name = f"...{name[-37:]}"  # 40 - 3 = 37
            else:
                truncated_name = name

            display = f"{truncated_name:<40}"  # Left-align and pad to 40

            if self.use_colors:
                display = ansi_utils.ANSIFormatter.format(
                    display,
                    *self.COMPONENT_STYLES['logger']
                )

        return display

    def _format_context(
        self,
        context: t.Dict[str, t.Any],
        prefix_len: int
    ) -> str:
        """Format context for console output."""
        # Remove TAG since it's in logger name
        ctx = {k: v for k, v in context.items() if k != 'TAG'}
        if not ctx:
            return ""

        indent = " " * (prefix_len - 3)  # for "==> "
        arrow = "==>"
        if self.use_colors:
            arrow = ansi_utils.ANSIFormatter.format(
                arrow,
                *self.COMPONENT_STYLES['context']
            )

        lines = []
        for k, v in ctx.items():
            line = f"{k}={v}"
            if self.use_colors:
                line = ansi_utils.ANSIFormatter.format(
                    line,
                    *self.COMPONENT_STYLES['context']
                )
            lines.append(f"\n{indent}{arrow} {line}")

        return "".join(lines)

    @staticmethod
    def _get_prefix_length(
        timestamp: str,
        logger: str,
        loglevel: str
    ) -> int:
        """Calculate prefix length for alignment."""
        return len(f"[{timestamp}] [{logger}] [{loglevel}] => ")

    def format(self, record: logging.LogRecord) -> str:
        context = self._extract_context(record)

        if self.is_console:
            return self._format_console(record, context)
        else:
            return self._format_json(record, context)

    def _format_console(
        self,
        record: logging.LogRecord,
        context: t.Dict[str, t.Any]
    ) -> str:
        """Format for console output with colors and alignment."""
        timestamp = self.formatTime(record, self.datefmt)
        logger_name = self._format_logger_with_tag(record.name, context)
        level_name = f"{record.levelname:<8}"
        message = record.getMessage()

        # Apply colors
        if self.use_colors:
            timestamp = ansi_utils.ANSIFormatter.format(
                timestamp,
                *self.COMPONENT_STYLES['timestamp']
            )
            level_name = ansi_utils.ANSIFormatter.format(
                level_name,
                *self.LEVEL_COLORS.get(record.levelno, ())
            )
            arrow = ansi_utils.ANSIFormatter.format(
                "=>",
                *self.COMPONENT_STYLES['arrow']
            )
        else:
            arrow = "=>"

        # Calculate alignment using plain text lengths
        plain_timestamp = self.formatTime(record, self.datefmt)
        plain_level = f"{record.levelname:<8}"
        prefix_len = self._get_prefix_length(
            plain_timestamp,
            "somnmind" + 32 * " ",  # Max 40-char logger
            plain_level
        )  # Use consistent 40-char logger
        indent = " " * (prefix_len - 3)

        # Handle multiline messages
        if '\n' in message:
            lines = message.split('\n')
            message_lines = [lines[0]]
            for line in lines[1:]:
                message_lines.append(f"{indent}{arrow} {line}")
            message = '\n'.join(message_lines)

        # Build log line
        log_line = (f"[{timestamp}] [{logger_name}] [{level_name}] {arrow} "
                    f"{message}")

        # Add context
        context_str = self._format_context(context, prefix_len)
        if context_str:
            log_line += context_str

        # Handle exceptions
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            if not log_line.endswith('\n'):
                log_line += '\n'

            exc_lines = [line for line in record.exc_text.split('\n') if
                         line.strip()]
            for line in exc_lines:
                colored_line = line
                if self.use_colors:
                    colored_line = ansi_utils.ANSIFormatter.format(
                        line,
                        ansi_utils.ANSIFormatter.FG.RED,
                        ansi_utils.ANSIFormatter.STYLE.DIM
                    )
                log_line += f"{indent}{arrow} {colored_line}\n"

        return log_line.rstrip('\n')

    def _format_json(
        self,
        record: logging.LogRecord,
        context: t.Dict[str, t.Any]
    ) -> str:
        """Format for file output as JSON."""
        data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        if context:
            data.update(context)

        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def _create_handler(target: LoggingTarget) -> logging.Handler:
    """Create appropriate handler based on target configuration."""
    if target.logname in ('stdout', 'stderr'):
        # Console handler
        stream = sys.stdout if target.logname == 'stdout' else sys.stderr
        handler = logging.StreamHandler(stream)
        formatter = ContextFormatter(is_console=True, use_colors=True)
    else:
        # File handler
        file_path = p.Path(target.logname)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if target.rotation:
            if target.rotation.size_based:
                max_bytes = target.rotation.size_based.max_size * 1024 * 1024
                handler = logging.handlers.RotatingFileHandler(
                    filename=str(file_path),
                    maxBytes=max_bytes,
                    backupCount=target.rotation.size_based.backup_count,
                    encoding='utf-8'
                )
            elif target.rotation.time_based:
                handler = logging.handlers.TimedRotatingFileHandler(
                    filename=str(file_path),
                    when='H',
                    interval=target.rotation.time_based.interval,
                    backupCount=target.rotation.time_based.backup_count,
                    encoding='utf-8'
                )
            else:
                handler = logging.FileHandler(
                    filename=str(file_path),
                    encoding='utf-8'
                )
        else:
            handler = logging.FileHandler(
                filename=str(file_path),
                encoding='utf-8'
            )

        formatter = ContextFormatter(is_console=False, use_colors=False)

    handler.setFormatter(formatter)
    handler.setLevel(_LEVEL_MAP.get(target.loglevel, logging.INFO))
    return handler


@utils.singleton
class NativeLoggingBackend(LoggerBackend):
    """Native logging backend with enhanced formatting."""

    def __init__(self):
        self._logger = logging.getLogger('somnmind')
        self._logger.setLevel(logging.DEBUG)
        self._handlers: t.List[logging.Handler] = []
        self._is_setup = False
        ansi_utils.ANSIFormatter.enable(
            ansi_utils.ANSIFormatter.supports_color()
        )

    def setup_handlers(self, targets: t.List[LoggingTarget]) -> None:
        if self._is_setup:
            return

        # Clear existing handlers
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        self._handlers.clear()

        # Create new handlers
        for target in targets:
            handler = _create_handler(target)
            self._logger.addHandler(handler)
            self._handlers.append(handler)

        self._logger.propagate = False
        self._is_setup = True

    def log(
        self,
        msg: str,
        /,
        level: LogLevel,
        **context: t.Any
    ) -> None:
        log_level = _LEVEL_MAP.get(level, logging.INFO)
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=log_level,
            fn='',
            lno=0,
            msg=msg,
            args=(),
            exc_info=None
        )

        # Add context to record
        for key, value in context.items():
            setattr(record, key, value)

        self._logger.handle(record)

    def sync(self) -> None:
        """Flush all handlers."""
        for handler in self._handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

    def close(self) -> None:
        """Close all handlers."""
        for handler in self._handlers:
            try:
                handler.close()
                self._logger.removeHandler(handler)
            except Exception:
                pass
        self._handlers.clear()
        self._is_setup = False


class NativeLogger(Logger):
    """Logger with rich ANSI color support."""

    def __init__(
        self,
        targets: t.Sequence[LoggingTarget] | None = None,
        initial_ctx: t.Dict[str, t.Any] | None = None,
    ):
        backend = NativeLoggingBackend()
        super().__init__(backend, targets, initial_ctx)

    @classmethod
    def enable_colors(cls, enabled: bool = True) -> None:
        """Enable or disable ANSI colors."""
        ansi_utils.ANSIFormatter.enable(enabled)

    @classmethod
    def supports_colors(cls) -> bool:
        """Check if terminal supports ANSI colors."""
        return ansi_utils.ANSIFormatter.supports_color()
