from __future__ import annotations

import typing as t

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from visflow import __prog__
from visflow.utils import Unset

if t.TYPE_CHECKING:
    from types import NoneType

    from pydantic.fields import _FieldInfoInputs

    T = t.TypeVar("T")


class PrototypexFieldInfo(FieldInfo):  # type: ignore[misc]
    """Extended FieldInfo with platform-specific default values and shareable
    support.

    This class extends Pydantic's FieldInfo to support platform-specific
    default values and factories, enabling generation of OS-specific
    configuration files.

    Attributes:
        linux_default: Default value for Linux systems
        linux_default_factory: Factory function for Linux default values
        windows_default: Default value for Windows systems
        windows_default_factory: Factory function for Windows default values
        system_default: Default value for system configuration
        system_default_factory: Factory function for system default
        system_path_type: Type of system path prefix (log, data, config, etc.)
        shareable: Whether this field can inherit from shared configuration
    """

    __slots__ = (
        *FieldInfo.__slots__,
        "linux_default",
        "linux_default_factory",
        "windows_default",
        "windows_default_factory",
        "system_default",
        "system_default_factory",
        "system_path_type",
        "shareable",
    )

    def __init__(
        self,
        linux_default: t.Any = PydanticUndefined,
        linux_default_factory: t.Callable[[], t.Any] | None = None,
        windows_default: t.Any = PydanticUndefined,
        windows_default_factory: t.Callable[[], t.Any] | None = None,
        system_default: t.Any = PydanticUndefined,
        system_default_factory: t.Callable[[], t.Any] | None = None,
        system_path_type: t.Literal["log", "data", "config", "cache", "runtime"] | None = None,
        shareable: bool = False,
        **kwargs: t.Unpack[_FieldInfoInputs],
    ) -> None:
        super().__init__(**kwargs)

        self.linux_default = linux_default
        self.linux_default_factory = linux_default_factory
        self.windows_default = windows_default
        self.windows_default_factory = windows_default_factory
        self.system_default = system_default
        self.system_default_factory = system_default_factory
        self.system_path_type = system_path_type
        self.shareable = shareable

    def get_platform_default(self, platform: str) -> t.Any:
        """Get the default value for the specified platform.

        Priority:
        1. Platform-specific factory (linux_default_factory/windows_default_factory)
        2. System path prefix + system_default (if system_path_type is set)
        3. Standard default

        Args:
            platform: Target platform ("linux" or "windows").

        Returns:
            Platform-specific default value, or PydanticUndefined if not set.
        """
        # Step 1: Try platform-specific factory/value
        platform_value = self._get_platform_specific_value(platform)

        # If found, join with system path if needed and return
        if platform_value is not PydanticUndefined and not isinstance(platform_value, Unset):
            return self._join_with_system_path(platform_value, platform)
        if isinstance(platform_value, Unset):
            return platform_value

        # Step 2: Try system_default with system_path_type
        if self.system_path_type is not None:
            system_value = self._get_system_default()
            if system_value is not PydanticUndefined:
                return self._join_with_system_path(system_value, platform)

        # Step 3: Fallback to standard default
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not PydanticUndefined:
            return self.default

        return PydanticUndefined

    def _get_platform_specific_value(self, platform: str) -> t.Any:
        """Get platform-specific value (factory or direct).

        Args:
            platform: Target platform.

        Returns:
            Platform-specific value or PydanticUndefined.
        """
        if platform == "linux":
            if self.linux_default_factory is not None:
                return self.linux_default_factory()
            if self.linux_default is not PydanticUndefined:
                return self.linux_default
        elif platform == "windows":
            if self.windows_default_factory is not None:
                return self.windows_default_factory()
            if self.windows_default is not PydanticUndefined:
                return self.windows_default

        return PydanticUndefined

    def _get_system_default(self) -> t.Any:
        """Get system default value (factory or direct).

        Returns:
            System default value or PydanticUndefined.
        """
        if self.system_default_factory is not None:
            return self.system_default_factory()
        if self.system_default is not PydanticUndefined:
            return self.system_default

        return PydanticUndefined

    def _get_system_path_prefix(self, platform: str) -> str | None:
        """Get the system path prefix for the specified platform.

        Args:
            platform: Target platform.

        Returns:
            System path prefix string, or None if system_path_type not set.
        """
        if self.system_path_type is None:
            return None

        app_name = __prog__

        # Linux FHS (Filesystem Hierarchy Standard) paths
        if platform == "linux":
            if self.system_path_type == "log":
                return f"${{HOME}}/.local/share/{app_name}/logs"
            if self.system_path_type == "data":
                return f"${{HOME}}/.local/share/{app_name}"
            if self.system_path_type == "config":
                return f"${{HOME}}/.config/{app_name}"
            if self.system_path_type == "cache":
                return f"${{HOME}}/.cache/{app_name}"
            if self.system_path_type == "runtime":
                return f"${{HOME}}/.cache/{app_name}/runtime"

        # Windows standard paths
        elif platform == "windows":
            if self.system_path_type == "log":
                return f"%PROGRAMDATA%\\{app_name}\\logs"
            if self.system_path_type == "data":
                return f"%PROGRAMDATA%\\{app_name}\\data"
            if self.system_path_type == "config":
                return f"%PROGRAMDATA%\\{app_name}\\config"
            if self.system_path_type == "cache":
                return f"%LOCALAPPDATA%\\{app_name}\\cache"
            if self.system_path_type == "runtime":
                return f"%TEMP%\\{app_name}"

        return None

    def _join_with_system_path(self, value: t.Any, platform: str) -> t.Any:
        """Join value with system path prefix if applicable.

        Args:
            value: Value to join.
            platform: Target platform.

        Returns:
            Joined path or original value.
        """
        if self.system_path_type is None or value is PydanticUndefined:
            return value

        prefix = self._get_system_path_prefix(platform)
        if prefix is None:
            return value

        # Only join string values
        if isinstance(value, str):
            separator = "/" if platform == "linux" else "\\"
            # Avoid double separators
            if not prefix.endswith(("/", "\\")):
                prefix += separator
            return prefix + value

        return value


def Field(
    default: T | Unset | NoneType = PydanticUndefined,
    *,
    default_factory: t.Callable[[], T] | None = None,
    linux_default: T | NoneType = PydanticUndefined,
    linux_default_factory: t.Callable[[], T] | None = None,
    windows_default: T | NoneType = PydanticUndefined,
    windows_default_factory: t.Callable[[], T] | None = None,
    system_default: T | NoneType = PydanticUndefined,
    system_default_factory: t.Callable[[], T] | None = None,
    system_path_type: t.Literal["log", "data", "config", "cache", "runtime"] | None = None,
    shareable: bool = False,
    alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    pattern: str | None = None,
    discriminator: t.Literal["type"] | str | None = None,
    examples: list[T] | None = None,
    **kwargs: t.Any,
) -> T:
    """Create a system-aware configuration field descriptor with shareable
    support.

    Args:
        default: Default value for all platforms
        default_factory: Factory function for default value
        linux_default: Default value specific to Linux
        linux_default_factory: Factory function for Linux default
        windows_default: Default value specific to Windows
        windows_default_factory: Factory function for Windows default
        system_default: Default value for system configuration
        system_default_factory: Factory function for system default
        system_path_type: Type of system path prefix
        shareable: Whether this field can inherit from shared configuration (NEW)
        alias: Field alias
        title: Field title
        description: Field description
        pattern: Regex pattern for string fields
        discriminator: Discriminator for union types
        examples: Example values for documentation
        **kwargs: Additional arguments passed to FieldInfo

    Returns:
        PrototypexFieldInfo instance
    """
    return PrototypexFieldInfo(
        default=default,
        default_factory=default_factory,
        linux_default=linux_default,
        linux_default_factory=linux_default_factory,
        windows_default=windows_default,
        windows_default_factory=windows_default_factory,
        system_default=system_default,
        system_default_factory=system_default_factory,
        system_path_type=system_path_type,
        shareable=shareable,
        alias=alias,
        title=title,
        description=description,
        pattern=pattern,
        discriminator=discriminator,
        examples=examples,
        **kwargs,
    )
