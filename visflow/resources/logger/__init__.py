from __future__ import annotations

import abc
import contextlib as cl
import contextvars as cvs
import functools as ft
import inspect
import time
import traceback as tb
import types
import typing as t

from visflow.resources.config.logging import LoggingConfig
from visflow.resources.logger.types import LoggingTarget, LogLevel

_context: cvs.ContextVar[t.Dict[str, t.Any]] = cvs.ContextVar(
    '_context', default={}
)
F = t.TypeVar('F', bound=t.Callable[..., t.Any])


class LogContext:
    @staticmethod
    def add(**ctx: t.Any) -> None:
        current_ctx = _context.get({})
        new_ctx = {**current_ctx, **ctx}
        _context.set(new_ctx)

    @staticmethod
    def get() -> t.Dict[str, t.Any]:
        return _context.get({})

    @staticmethod
    def clear() -> None:
        _context.set({})


class LoggerBackend(abc.ABC):

    @abc.abstractmethod
    def setup_handlers(self, targets: t.List[LoggingTarget]) -> None:
        pass

    @abc.abstractmethod
    def log(
        self,
        msg: str, /,
        level: LogLevel,
        **context: t.Any
    ) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class _LoggerContext:

    def __init__(
        self,
        backend: LoggerBackend,
        initial_ctx: t.Dict[str, t.Any] | None = None
    ):
        self._backend = backend
        self._context = initial_ctx or {}

    def with_context(self, /, **kwargs: t.Any) -> _LoggerContext:
        new_context = _LoggerContext(self._backend)
        new_context._context = {**self._context, **kwargs}
        return new_context

    def with_tag(self, tag: str, /) -> _LoggerContext:
        return self.with_context(TAG=tag)

    def with_request_id(self, request_id: str, /) -> _LoggerContext:
        return self.with_context(REQUEST_ID=request_id)

    def with_user_id(self, user_id: str, /) -> _LoggerContext:
        return self.with_context(USER_ID=user_id)

    def log(self, msg: str, /, level: LogLevel, **kwargs: t.Any) -> None:
        ctx = {**self._context, **(kwargs or {})}
        self._backend.log(msg, level, **ctx)

    def debug(self, msg: str, /, **kwargs: t.Any) -> None:
        return self.log(msg, level='debug', **kwargs)

    def info(self, msg: str, /, **kwargs: t.Any) -> None:
        return self.log(msg, level='info', **kwargs)

    def warning(self, msg: str, /, **kwargs: t.Any) -> None:
        return self.log(msg, level='warning', **kwargs)

    def error(self, msg: str, /, **kwargs: t.Any) -> None:
        return self.log(msg, level='error', **kwargs)

    def critical(self, msg: str, /, **kwargs: t.Any) -> None:
        return self.log(msg, level='critical', **kwargs)

    @cl.contextmanager
    def catch(
        self,
        msg: str, /,
        exc:
        type[BaseException] | t.Tuple[type[BaseException], ...] = Exception,
        excl_exc:
        type[BaseException] | t.Tuple[type[BaseException], ...] = ()
    ) -> t.Generator[None, None, None]:
        try:
            yield
        except exc as e:
            if isinstance(e, excl_exc):
                raise
            self.error(
                f"{msg}: {e}",
                exception_type=type(e).__name__,
                exception=e,
                traceback=tb.format_exc()
            )
            raise

    def log_method(
        self,
        msg: str | None = None,
        level: LogLevel = 'info',
        *,
        exc:
        type[BaseException] | t.Tuple[type[BaseException], ...] = Exception,
        excl_exc:
        type[BaseException] | t.Tuple[type[BaseException], ...] = (),
        logargs: bool = False,
        logres: bool = False,
        logdur: bool = True,
        incl_ctx: bool = True,
        success_level: LogLevel = 'info',
        error_level: LogLevel = 'error',
        pre_exec: bool = True,
        post_exec: bool = True,
    ) -> t.Callable[[F], F]:

        def decorator(fn: F) -> t.Any:
            @ft.wraps(fn)
            def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
                func_name = fn.__name__
                module_name = getattr(fn, '__module__', 'unknown')
                class_name = None

                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__
                    full_name = f"{class_name}.{func_name}"
                else:
                    full_name = func_name

                LogContext.clear()

                base_ctx = {
                    'function': func_name,
                    'module': module_name,
                    'full_name': full_name,
                }

                if class_name:
                    base_ctx['class'] = class_name

                if logargs:
                    sig = inspect.signature(fn)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    filtered_args = {
                        k: v for k, v in bound_args.arguments.items()
                        if k not in ('self', 'cls')
                    }
                    base_ctx['arguments'] = filtered_args

                start_time = time.time()
                if pre_exec:
                    pre_msg = msg or f"Executing {full_name}"
                    self.log(
                        pre_msg,
                        level=level,
                        **base_ctx,
                        execution_stage='pre'
                    )

                try:
                    result = fn(*args, **kwargs)

                    decorator_ctx = LogContext.get() if incl_ctx else {}

                    if post_exec:
                        end_time = time.time()
                        duration = end_time - start_time

                        success_context = {**base_ctx, **decorator_ctx}

                        if logdur:
                            success_context['duration_ms'] = round(
                                duration * 1000, 2
                            )

                        if logres:
                            try:
                                if hasattr(result, '__dict__'):
                                    success_context['result'] = str(result)
                                elif isinstance(
                                    result,
                                    (str, int, float, bool, list, dict)
                                ):
                                    success_context['result'] = result
                                else:
                                    success_context['result'] = str(result)
                            except Exception:
                                success_context['result'] = '<unserializable>'

                        success_msg = msg or f"Completed {full_name}"
                        self.log(
                            success_msg,
                            level=success_level,
                            **success_context,
                            execution_stage='post',
                            status='success'
                        )

                    return result

                except exc as e:
                    if isinstance(e, excl_exc):
                        raise

                    decorator_ctx = LogContext.get() if incl_ctx else {}

                    end_time = time.time()
                    duration = end_time - start_time

                    error_context = {
                        **base_ctx,
                        **decorator_ctx,
                        'exception_type': type(e).__name__,
                        'exception': str(e),
                        'traceback': tb.format_exc(),
                        'execution_stage': 'error',
                        'status': 'error'
                    }

                    if logdur:
                        error_context['duration_ms'] = (
                            round(duration * 1000, 2)
                        )

                    error_msg = msg or f"Failed {full_name}: {e}"
                    self.log(
                        error_msg,
                        level=error_level,
                        **error_context
                    )

                    raise
                finally:
                    LogContext.clear()

            return wrapper

        return decorator


class BaseLogger(_LoggerContext):
    def __init__(
        self,
        backend: LoggerBackend,
        targets: t.Sequence[LoggingTarget] | None = None,
        initial_ctx: t.Dict[str, t.Any] | None = None
    ):
        super().__init__(backend, initial_ctx)
        self._targets = list(targets or [])
        self._backend.setup_handlers(self._targets)

    @property
    def targets(self) -> t.List[LoggingTarget]:
        return self._targets

    def add_target(self, *target: LoggingTarget) -> None:
        self._targets.extend(target)
        self._backend.setup_handlers(self._targets)

    def remove_target(self, logname: str) -> None:
        self._targets = [
            t for t in self._targets if t.logname != logname
        ]
        self._backend.setup_handlers(self._targets)

    def clear_targets(self) -> None:
        self._targets = []
        self._backend.setup_handlers(self._targets)

    def sync(self) -> None:
        self._backend.sync()

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> t.Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None, /
    ) -> t.Literal[False]:
        self.close()
        return False


class Logger(BaseLogger):
    def __init__(self, config: LoggingConfig):
        self.config = config

        if self.config.backend == 'native':
            import visflow.resources.logger.native as native
            backend = native.NativeLoggingBackend()  # type: LoggerBackend
        elif self.config.backend == 'loguru':
            import visflow.resources.logger.loguru as loguru
            backend = loguru.LoguruBackend()
        else:
            raise ValueError(
                f"Unsupported logging backend: {self.config.backend}",
            )

        super().__init__(
            backend=backend,
            initial_ctx=self.config.extra_context
        )
        self.add_target(
            LoggingTarget(logname='stdout', loglevel=self.config.loglevel)
        )
