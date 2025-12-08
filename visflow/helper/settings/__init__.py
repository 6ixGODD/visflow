from __future__ import annotations

import json
import os
import pathlib
import typing as t
import weakref

from pydantic import BaseModel as PydBaseModel
from pydantic import ConfigDict
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings as PydBaseSettings

from visflow.exceptions import InvalidConfigurationError
from visflow.exceptions import RequiredModuleNotFoundError
from visflow.helper.settings.fields import Field
from visflow.helper.settings.fields import PrototypexFieldInfo
from visflow.utils import Unset
from visflow.utils import flatten_dict


class BaseModel(PydBaseModel):
    """Base model class with common configuration for all models.

    This class sets default configurations for Pydantic models used in
    the Visflow project, ensuring consistent behavior across all derived
    models.
    """

    model_config: t.ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        extra="ignore",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
    )

    def __hash__(self) -> int:
        """Generate a hash based on the model's serializable representation."""
        serl_dict = self.model_dump()
        serl_json = json.dumps(serl_dict, sort_keys=True)
        return hash(serl_json)

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{field_name}={getattr(self, field_name)!r}"
            for field_name in self.__class__.model_fields  # type: ignore
        )
        return f"{self.__class__.__name__}({field_reprs})"

    def as_dict(self) -> dict[str, t.Any]:
        """Get a dictionary representation of the model."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, _data: dict[str, t.Any]) -> t.Self:
        """Create a model instance from a dictionary."""
        import this  # noqa

        raise SystemError(
            "Why you call from_dict on BaseModel? This is NOT allowed here! !!  "
            "DONT use a dict to create BaseModel directly, use Settings instead!  "
            "PLEASE FIX YOUR CODE!"
        )


# ============================================================================
# Global Registry for Shareable Models
# ============================================================================

# Maps SharableBase type -> weakref to instance
_GLOBAL_SHAREABLE_REGISTRY: dict[type, weakref.ref] = {}


def _register_shareable(instance: SharableBase) -> None:
    """Register a shareable instance in the global registry."""
    _GLOBAL_SHAREABLE_REGISTRY[type(instance)] = weakref.ref(instance)


def _get_shareable(shareable_type: type) -> SharableBase | None:
    """Get a registered shareable instance by type."""
    weak_ref = _GLOBAL_SHAREABLE_REGISTRY.get(shareable_type)
    if weak_ref is None:
        return None

    instance = weak_ref()
    if instance is None:
        # Garbage collected, clean up
        del _GLOBAL_SHAREABLE_REGISTRY[shareable_type]
        return None

    return instance


def _clear_shareable_registry() -> None:
    """Clear the global shareable registry."""
    _GLOBAL_SHAREABLE_REGISTRY.clear()


# ============================================================================
# Shareable Models
# ============================================================================


class SharableBase(BaseModel):
    """Base class for shareable configuration models.

    Models inheriting from this class can be registered and shared
    across different parts of the configuration.
    """

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{field_name}={getattr(self, field_name)!r}"
            for field_name in self.__class__.model_fields  # type: ignore
        )
        return f"{self.__class__.__name__}<sharable>({field_reprs})"


class SharedBase(BaseModel):
    """Base class for models that contain shareable configurations.

    Models inheriting from this class will be scanned for SharableBase
    instances during Settings initialization.
    """

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{field_name}={getattr(self, field_name)!r}"
            for field_name in self.__class__.model_fields  # type: ignore
        )
        return f"{self.__class__.__name__}<shared>({field_reprs})"


class Settings(PydBaseSettings):
    """Base settings class with YAML/JSON/dotenv export capabilities."""

    def model_post_init(self, __context: t.Any) -> None:
        """Post-initialization hook to handle shareable configurations.

        Steps:
        1. Clear the global registry (for fresh start)
        2. Find all SharedBase instances
        3. Scan SharedBase instances for SharableBase children and register
           them globally
        4. Find all fields with annotation SharableBase | None and value None
        5. Fill those fields from the global registry
        """
        super().model_post_init(__context)

        # Step 1: Clear global registry for this config instance
        _clear_shareable_registry()

        # Step 2 & 3: Scan SharedBase and register SharableBase instances globally
        self._scan_and_register_shareables(self)

        # Step 4 & 5: Fill shareable fields from global registry
        self._fill_shareable_fields(self)

    def _scan_and_register_shareables(self, obj: t.Any) -> None:
        """Recursively find SharedBase instances and register their
        SharableBase children globally.

        Args:
            obj: Object to scan
        """
        if obj is None:
            return

        # If this is a SharedBase instance, scan its fields for SharableBase
        if isinstance(obj, SharedBase):
            for field_name in obj.__class__.model_fields:  # type: ignore
                field_value = getattr(obj, field_name)
                if isinstance(field_value, SharableBase):
                    # Register in global registry
                    _register_shareable(field_value)

        # Continue recursing for nested structures
        if isinstance(obj, PydBaseModel):
            for field_name in obj.__class__.model_fields:
                field_value = getattr(obj, field_name)
                if field_value is not None:
                    self._scan_and_register_shareables(field_value)

        elif isinstance(obj, dict):
            for value in obj.values():
                self._scan_and_register_shareables(value)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._scan_and_register_shareables(item)

    def _fill_shareable_fields(self, obj: t.Any) -> None:
        """Recursively find and fill None fields that have SharableBase types.

        Args:
            obj: Object to process
        """
        if obj is None:
            return

        # Process PydBaseModel instances
        if isinstance(obj, PydBaseModel) and not isinstance(obj, SharedBase):
            for field_name, field_info in obj.__class__.model_fields.items():
                field_value = getattr(obj, field_name)

                # If field is None, check if it's a shareable type
                if field_value is None:
                    shareable_type = self._get_shareable_type(field_info)
                    if shareable_type is not None:
                        # Get from global registry
                        shareable_instance = _get_shareable(shareable_type)
                        if shareable_instance is not None:
                            setattr(obj, field_name, shareable_instance)
                            continue

                # Recurse into non-None fields
                if field_value is not None:
                    self._fill_shareable_fields(field_value)

        # Recurse into dictionaries
        elif isinstance(obj, dict):
            for value in obj.values():
                self._fill_shareable_fields(value)

        # Recurse into lists/tuples
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._fill_shareable_fields(item)

    def _get_shareable_type(self, field_info: t.Any) -> type[SharableBase] | None:
        """Extract SharableBase type from field annotation if it's shareable.

        Args:
            field_info: Field info object

        Returns:
            The SharableBase type if field is marked shareable and type is
            SharableBase | None
        """
        annotation = field_info.annotation

        # Check if field is marked as shareable
        is_marked_shareable = False
        if hasattr(field_info, "shareable"):
            is_marked_shareable = bool(field_info.shareable)
        elif hasattr(field_info, "json_schema_extra"):
            json_schema_extra = field_info.json_schema_extra
            if isinstance(json_schema_extra, dict):
                is_marked_shareable = json_schema_extra.get("shareable", False)

        # If not marked as shareable, skip
        if not is_marked_shareable:
            return None

        # Direct type check (SharableBase)
        if isinstance(annotation, type):
            try:
                if issubclass(annotation, SharableBase):
                    return annotation
            except TypeError:
                pass

        # Handle Optional[SharableBase] or SharableBase | None
        if hasattr(annotation, "__args__"):
            for arg in annotation.__args__:
                if arg is not type(None) and isinstance(arg, type):
                    try:
                        if issubclass(arg, SharableBase):
                            return arg
                    except TypeError:
                        pass

        return None

    # ============================================================================
    # Public Methods - Loading
    # ============================================================================

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | os.PathLike[str]) -> t.Self:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Settings instance.

        Raises:
            RequiredModuleNotFoundError: If PyYAML is not installed.
        """
        try:
            import yaml

            data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
            return cls.model_validate(data, strict=False)
        except ImportError as e:
            raise RequiredModuleNotFoundError("PyYAML is required to load YAML files.") from e

    # ============================================================================
    # Public Methods - Export
    # ============================================================================

    def serl(self, include_none: bool = False) -> dict[str, t.Any]:
        """Get a serializable version of the model dump.

        Args:
            include_none: If True, include fields with None values.

        Returns:
            Dictionary with only serializable values.
        """
        raw_dump = self.model_dump()
        return self._clean_dict(raw_dump, include_none=include_none)

    def to_example_yaml(
        self,
        fpath: str | pathlib.Path | os.PathLike[str],
        exclude_unset: bool = False,
        exclude_none: bool = False,
    ) -> None:
        """Export example configuration to YAML file with field metadata,
        examples, and constraints as comments.

        Args:
            fpath: Path to the output YAML file.
            exclude_unset: If True, exclude fields with Unset values and empty
                nested models.
            exclude_none: If True, exclude fields with None values.
        """
        metadata = self._collect_field_metadata()
        data = self.serl(include_none=True)

        # Clean unset and none values
        if exclude_unset:
            data = self._clean_unset_recursive(data)
        if exclude_none:
            data = self._clean_none_recursive(data)

        with pathlib.Path(fpath).open("w", encoding="utf-8") as f:
            f.write("# Example Configuration File\n")
            f.write("# This file demonstrates all available configuration options.\n")
            f.write("# Copy and modify as needed for your environment.\n\n")
            self._write_yaml_with_metadata(f, data, metadata, "")

    def to_system_yaml(
        self,
        fpath: str | pathlib.Path | os.PathLike[str],
        platform: t.Literal["linux", "windows"] | None = None,
    ) -> None:
        """Export system configuration to YAML file with platform-specific
        defaults.

        This method generates a system configuration file using
        platform-specific default values defined in PrototypexFieldInfo
        descriptors. Fields with Unset values are omitted. Empty nested models
        (all fields Unset) are also omitted.

        Args:
            fpath: Path to the output YAML file
            platform: Target platform ("linux" or "windows"). If None, uses
                current platform.
        """
        if platform is None:
            platform = "linux" if os.name != "nt" else "windows"

        # Build data with platform-specific defaults
        data = self._build_platform_data(self.__class__, platform)

        # Clean Unset values and empty dicts
        data = self._clean_unset_values(data)

        # Write YAML without comments
        with pathlib.Path(fpath).open("w", encoding="utf-8") as f:
            self._write_system_yaml_header(f, platform)
            self._write_yaml_dict(f, data)

    def to_example_dotenv(self, fpath: str | pathlib.Path | os.PathLike[str]) -> None:
        """Export example configuration to .env file with field metadata,
        examples, and constraints as comments.

        Args:
            fpath: Path to the output .env file.
        """
        metadata = self._collect_field_metadata()
        self._write_dotenv_file(fpath, metadata, is_example=True)

    def to_system_dotenv(
        self,
        fpath: str | pathlib.Path | os.PathLike[str],
        platform: t.Literal["linux", "windows"] | None = None,
    ) -> None:
        """Export system configuration to .env file with platform-specific
        defaults.

        Args:
            fpath: Path to the output .env file.
            platform: Target platform ("linux" or "windows"). If None, uses
                current platform.
        """
        if platform is None:
            platform = "linux" if os.name != "nt" else "windows"

        # Build data with platform-specific defaults
        data = self._build_platform_data(self.__class__, platform)
        data = self._clean_unset_values(data)

        # Flatten the data
        sep = self.model_config.get("env_nested_delimiter") or "__"
        prefix = self.model_config.get("env_prefix") or "AUDEX__"
        if prefix.endswith(sep):
            prefix = prefix[: -len(sep)]

        with pathlib.Path(fpath).open("w", encoding="utf-8") as f:
            f.write("# System Configuration (.env)\n")
            f.write(f"# Platform: {platform}\n")
            f.write("# This file is generated automatically.\n\n")

            for top_key, top_value in data.items():
                section_data = {top_key: top_value}
                flattened = flatten_dict(section_data, sep=sep)

                f.write(f"# {'-' * 70}\n")
                f.write(f"# {top_key.upper()}\n")
                f.write(f"# {'-' * 70}\n\n")

                for key, value in flattened.items():
                    env_key = f"{prefix}{sep}{key.upper()}"
                    if value is None:
                        f.write(f"{env_key}=\n")
                    else:
                        formatted_value = self._format_env_value(value)
                        f.write(f"{env_key}={formatted_value}\n")
                f.write("\n")

    # ============================================================================
    # Private Methods - Field Metadata Collection
    # ============================================================================

    def _format_type_annotation(self, annotation: t.Any) -> str:
        """Format a type annotation as a readable string.

        Args:
            annotation: The type annotation to format.

        Returns:
            Formatted type string.
        """
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        # Handle typing generics
        annotation_str = str(annotation)
        # Clean up typing module prefixes
        return annotation_str.replace("typing.", "")

    def _extract_constraints(self, field_info: t.Any) -> dict[str, t.Any]:
        """Extract validation constraints from a field.

        Args:
            field_info: The field info object.

        Returns:
            Dictionary of constraint name to value.
        """
        constraints = {}
        constraint_names = ["gt", "ge", "lt", "le", "min_length", "max_length", "pattern"]

        for name in constraint_names:
            value = getattr(field_info, name, None)
            if value is not None:
                constraints[name] = value

        return constraints

    # ============================================================================
    # Private Methods - YAML Writing with Metadata
    # ============================================================================

    def _write_yaml_with_metadata(
        self,
        f: t.TextIO,
        data: dict[str, t.Any],
        metadata: dict[str, dict[str, t.Any]],
        prefix: str,
        indent: int = 0,
    ) -> None:
        """Write YAML with metadata comments.

        Args:
            f: File object.
            data: Data dictionary to write.
            metadata: Field metadata.
            prefix: Field path prefix.
            indent: Indentation level.
        """
        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key
            field_meta = metadata.get(field_path, {})

            # Write metadata comments
            self._write_field_metadata_comment(f, field_meta, indent)

            # Write the value
            self._write_yaml_value_with_examples(f, key, value, field_meta, field_path, metadata, indent)

    def _write_field_metadata_comment(
        self,
        f: t.TextIO,
        field_meta: dict[str, t.Any],
        indent: int,
    ) -> None:
        """Write field metadata as comments.

        Args:
            f: File object.
            field_meta: Field metadata dictionary.
            indent: Indentation level.
        """
        indent_str = "  " * indent

        # Description (if exists)
        if field_meta.get("description"):
            description = field_meta["description"]
            # Handle multi-line descriptions
            if "\n" in description:
                f.write(f"{indent_str}# Description:\n")
                for line in description.split("\n"):
                    f.write(f"{indent_str}#   {line}\n")
            else:
                f.write(f"{indent_str}# {description}\n")

        # Build metadata line
        meta_parts = []

        # Type
        if field_meta.get("type"):
            meta_parts.append(f"Type: {field_meta['type']}")

        # Default value
        if field_meta.get("default") is not None:
            default_val = field_meta["default"]
            if not isinstance(default_val, PydBaseModel):
                meta_parts.append(f'Default: "{default_val!s}"')

        # Constraints
        constraints = field_meta.get("constraints", {})
        if constraints:
            constraint_strs = []
            if "ge" in constraints:
                constraint_strs.append(f">={constraints['ge']}")
            if "gt" in constraints:
                constraint_strs.append(f">{constraints['gt']}")
            if "le" in constraints:
                constraint_strs.append(f"<={constraints['le']}")
            if "lt" in constraints:
                constraint_strs.append(f"<{constraints['lt']}")
            if "min_length" in constraints:
                constraint_strs.append(f"min_len={constraints['min_length']}")
            if "max_length" in constraints:
                constraint_strs.append(f"max_len={constraints['max_length']}")
            if "pattern" in constraints:
                constraint_strs.append(f"pattern={constraints['pattern']}")

            if constraint_strs:
                meta_parts.append(f"Constraints: {', '.join(constraint_strs)}")

        # Write metadata line if there's any info
        if meta_parts:
            f.write(f"{indent_str}# {' | '.join(meta_parts)}\n")

    def _write_yaml_value_with_examples(
        self,
        f: t.TextIO,
        key: str,
        value: t.Any,
        field_meta: dict[str, t.Any],
        field_path: str,
        metadata: dict[str, dict[str, t.Any]],
        indent: int,
    ) -> None:
        """Write YAML value with examples as comments.

        Args:
            f: File object.
            key: Field key.
            value: Field value.
            field_meta: Field metadata.
            field_path: Field path.
            metadata: All field metadata.
            indent: Indentation level.
        """
        indent_str = "  " * indent
        examples = field_meta.get("examples")

        # Use first example as the value if available
        if examples and isinstance(examples, list) and len(examples) > 0:
            example_value = examples[0]
            if hasattr(example_value, "example"):
                example_value = example_value.example

            # Write the example value
            if isinstance(value, dict):
                f.write(f"{indent_str}{key}:\n")
                if isinstance(example_value, dict):
                    for sub_key, sub_value in example_value.items():
                        yaml_val = self._yaml_repr(sub_value)
                        f.write(f"{indent_str}  {sub_key}: {yaml_val}\n")
                else:
                    self._write_yaml_dict(f, value, indent + 1)
            else:
                yaml_value = self._yaml_repr(example_value)
                f.write(f"{indent_str}{key}: {yaml_value}\n")

            # Write remaining examples as comments (more compact)
            if len(examples) > 1:
                other_examples = []
                for ex in examples[1:]:
                    ex_val = ex.example if hasattr(ex, "example") else ex
                    other_examples.append(self._yaml_repr(ex_val))
                f.write(f"{indent_str}# Other examples: {', '.join(other_examples)}\n")
        else:
            # No examples, write value normally
            if value is None:
                f.write(f"{indent_str}{key}: ~\n")
            elif isinstance(value, dict):
                f.write(f"{indent_str}{key}:\n")
                self._write_yaml_with_metadata(f, value, metadata, field_path, indent + 1)
            elif isinstance(value, list):
                f.write(f"{indent_str}{key}:\n")
                self._write_yaml_list(f, value, indent + 1)
            else:
                yaml_value = self._yaml_repr(value)
                f.write(f"{indent_str}{key}: {yaml_value}\n")

        f.write("\n")

    def _collect_field_metadata(
        self,
        model: type[BaseModel] | None = None,
        prefix: str = "",
    ) -> dict[str, dict[str, t.Any]]:
        """Recursively collect field metadata including type, constraints,
        description, default values, and examples.

        Args:
            model: The pydantic model to collect metadata from.
            prefix: The prefix for nested field paths.

        Returns:
            A dictionary mapping field paths to their metadata.
        """
        if model is None:
            model = self.__class__

        metadata = {}
        for field_name, field_info in model.model_fields.items():
            field_path = f"{prefix}.{field_name}" if prefix else field_name

            # Get default value
            default_value = None
            if field_info.default is not PydanticUndefined:
                default_value = field_info.default
            elif field_info.default_factory is not None:
                try:
                    # Try to call default_factory to get the default value
                    default_value = field_info.default_factory()
                except Exception:
                    # If it fails, just note that it has a factory
                    default_value = "<factory>"

            # Collect field metadata
            field_meta = {
                "description": field_info.description,
                "type": self._format_type_annotation(field_info.annotation),
                "default": default_value,
                "constraints": self._extract_constraints(field_info),
                "examples": getattr(field_info, "examples", None),
            }

            metadata[field_path] = field_meta

            # Recurse into nested models
            if hasattr(field_info.annotation, "model_fields"):
                nested_metadata = self._collect_field_metadata(field_info.annotation, prefix=field_path)
                metadata.update(nested_metadata)

        return metadata

    # ============================================================================
    # Private Methods - Dotenv Writing
    # ============================================================================

    def _write_dotenv_file(
        self,
        fpath: str | pathlib.Path | os.PathLike[str],
        metadata: dict[str, dict[str, t.Any]],
        is_example: bool = True,
    ) -> None:
        """Write dotenv file with metadata.

        Args:
            fpath: File path.
            metadata: Field metadata.
            is_example: Whether this is an example file.
        """
        sep = self.model_config.get("env_nested_delimiter") or "__"
        prefix = self.model_config.get("env_prefix") or "AUDEX__"

        if prefix.endswith(sep):
            prefix = prefix[: -len(sep)]

        serializable_data = self.serl(include_none=True)

        grouped_keys = {}
        for top_level_key in serializable_data:
            section_data = {top_level_key: serializable_data[top_level_key]}
            flattened = flatten_dict(section_data, sep=sep)
            grouped_keys[top_level_key] = flattened

        with pathlib.Path(fpath).open("w", encoding="utf-8") as f:
            if is_example:
                f.write("# Example Environment Configuration File\n")
                f.write("# Copy this file to '.env' and modify the values as needed.\n\n")
            else:
                f.write("# Environment Configuration File\n\n")

            for top_key, flattened in grouped_keys.items():
                f.write(f"# {'=' * 70}\n")
                f.write(f"# {top_key.upper()}\n")
                f.write(f"# {'=' * 70}\n\n")

                for key, value in flattened.items():
                    env_key = f"{prefix}{sep}{key.upper()}"
                    field_path = key.replace(sep, ".")
                    field_meta = metadata.get(field_path, {})

                    # Write field metadata
                    if field_meta.get("description"):
                        f.write(f"# {field_meta['description']}\n")
                    if field_meta.get("type"):
                        f.write(f"# Type: {field_meta['type']}\n")

                    constraints = field_meta.get("constraints", {})
                    if constraints:
                        constraint_parts = []
                        for c_name, c_value in constraints.items():
                            constraint_parts.append(f"{c_name}={c_value}")
                        f.write(f"# Constraints: {', '.join(constraint_parts)}\n")

                    # Handle examples
                    examples = field_meta.get("examples")
                    if examples and isinstance(examples, list) and len(examples) > 0:
                        example_value = examples[0]
                        if hasattr(example_value, "example"):
                            example_value = example_value.example

                        formatted_value = self._format_env_value(example_value)
                        f.write(f"{env_key}={formatted_value}\n")

                        # Write remaining examples as comments
                        if len(examples) > 1:
                            f.write("# Other examples:\n")
                            for ex in examples[1:]:
                                ex_val = ex.example if hasattr(ex, "example") else ex
                                f.write(f"#   {self._format_env_value(ex_val)}\n")
                    else:
                        # No examples
                        if value is None:
                            f.write(f"# {env_key}=\n")
                        else:
                            formatted_value = self._format_env_value(value)
                            f.write(f"{env_key}={formatted_value}\n")

                    f.write("\n")

    # ============================================================================
    # Private Methods - Field Introspection (kept for compatibility)
    # ============================================================================

    def _collect_field_desc(
        self,
        model: type[BaseModel] | None = None,
        prefix: str = "",
    ) -> dict[str, str]:
        """Recursively collect field descriptions from the model and nested
        models.

        Args:
            model: The pydantic model to collect descriptions from.
            prefix: The prefix for nested field paths.

        Returns:
            A dictionary mapping field paths to their descriptions.
        """
        if model is None:
            model = self.__class__

        descriptions = {}
        for field_name, field_info in model.model_fields.items():
            field_path = f"{prefix}.{field_name}" if prefix else field_name

            if field_info.description:
                descriptions[field_path] = field_info.description

            if hasattr(field_info.annotation, "model_fields"):
                nested_descriptions = self._collect_field_desc(field_info.annotation, prefix=field_path)
                descriptions.update(nested_descriptions)

        return descriptions

    # ============================================================================
    # Private Methods - Serialization
    # ============================================================================

    def _serl_value(self, value: t.Any, include_none: bool = False) -> t.Any:
        """Serialize a value, handling special cases like callables.

        Args:
            value: The value to serialize.
            include_none: If True, preserve None values in nested structures.

        Returns:
            Serialized value or None if not serializable.
        """
        if value is None:
            return value if include_none else None

        if isinstance(value, os.PathLike):
            try:
                return os.fspath(value)
            except Exception:
                return None

        if callable(value):
            try:
                return self._serl_value(value(), include_none=include_none)
            except Exception:
                return None

        if isinstance(value, PydBaseModel):
            return value.model_dump()

        serialized: list[t.Any] | dict[str, t.Any]

        if isinstance(value, (list | tuple)):
            serialized = []
            for item in value:
                serialized_item = self._serl_value(item, include_none=include_none)
                if include_none or serialized_item is not None:
                    serialized.append(serialized_item)
            return serialized if serialized or include_none else None

        if isinstance(value, dict):
            serialized = {}
            for k, v in value.items():
                serialized_v = self._serl_value(v, include_none=include_none)
                if include_none or serialized_v is not None:
                    serialized[k] = serialized_v
            return serialized if serialized or include_none else None

        if isinstance(value, (str | int | float | bool)):
            return value

        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return None

    def _clean_dict(self, data: dict[str, t.Any], include_none: bool = False) -> dict[str, t.Any]:
        """Recursively clean a dictionary to remove non-serializable values.

        Args:
            data: Dictionary to clean.
            include_none: If True, include fields with None values.

        Returns:
            Cleaned dictionary.
        """
        cleaned: dict[str, t.Any] = {}
        for key, value in data.items():
            if value is None:
                if include_none:
                    cleaned[key] = None
            else:
                serialized = self._serl_value(value, include_none=include_none)
                if serialized is not None or include_none:
                    cleaned[key] = serialized
        return cleaned

    def _clean_unset_recursive(self, data: t.Any) -> t.Any:
        """Recursively remove Unset values and empty nested structures.

        Args:
            data: Data structure to clean.

        Returns:
            Cleaned data structure with Unset values and empty dicts removed.
        """
        if isinstance(data, Unset) or (isinstance(data, str) and data == "<UNSET>"):
            return None

        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Skip Unset values
                if isinstance(value, Unset) or (isinstance(value, str) and value == "<UNSET>"):
                    continue

                cleaned_value = self._clean_unset_recursive(value)

                # Skip None (which indicates Unset was found)
                if cleaned_value is None:
                    continue

                # Skip empty dicts (all nested fields were Unset)
                if isinstance(cleaned_value, dict) and not cleaned_value:
                    continue

                # Skip empty lists
                if isinstance(cleaned_value, list) and not cleaned_value:
                    continue

                cleaned[key] = cleaned_value

            # Return None if dict is empty (all fields were Unset)
            return cleaned if cleaned else None

        if isinstance(data, list):
            cleaned_list = []
            for item in data:
                if isinstance(item, Unset) or (isinstance(item, str) and item == "<UNSET>"):
                    continue

                cleaned_item = self._clean_unset_recursive(item)
                if cleaned_item is not None:
                    # Don't add empty dicts or lists
                    if isinstance(cleaned_item, (dict | list)) and not cleaned_item:
                        continue
                    cleaned_list.append(cleaned_item)

            return cleaned_list if cleaned_list else None

        return data

    def _clean_none_recursive(self, data: t.Any) -> t.Any:
        """Recursively remove None values and empty nested structures.

        Args:
            data: Data structure to clean.

        Returns:
            Cleaned data structure with None values removed.
        """
        if data is None:
            return None

        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if value is None:
                    continue

                cleaned_value = self._clean_none_recursive(value)

                if cleaned_value is None:
                    continue

                # Skip empty dicts
                if isinstance(cleaned_value, dict) and not cleaned_value:
                    continue

                # Skip empty lists
                if isinstance(cleaned_value, list) and not cleaned_value:
                    continue

                cleaned[key] = cleaned_value

            return cleaned if cleaned else None

        if isinstance(data, list):
            cleaned_list = []
            for item in data:
                if item is None:
                    continue

                cleaned_item = self._clean_none_recursive(item)
                if cleaned_item is not None:
                    if isinstance(cleaned_item, (dict | list)) and not cleaned_item:
                        continue
                    cleaned_list.append(cleaned_item)

            return cleaned_list if cleaned_list else None

        return data

    # ============================================================================
    # Private Methods - System YAML Generation
    # ============================================================================

    def _build_platform_data(
        self,
        model: type[PydBaseModel],
        platform: str,
        prefix: str = "",
    ) -> dict[str, t.Any]:
        """Recursively build configuration data with platform-specific
        defaults.

        Args:
            model: Pydantic model to process.
            platform: Target platform.
            prefix: Field path prefix.

        Returns:
            Dictionary with platform-specific default values.
        """

        result = {}

        for field_name, field_info in model.model_fields.items():
            field_path = f"{prefix}.{field_name}" if prefix else field_name

            # Handle nested models
            if hasattr(field_info.annotation, "model_fields"):
                nested_data = self._build_platform_data(
                    field_info.annotation,  # type: ignore
                    platform,
                    field_path,
                )
                # Only add if nested model has content
                if nested_data:
                    result[field_name] = nested_data
                continue

            # Get value based on field type
            if isinstance(field_info, PrototypexFieldInfo):
                value = field_info.get_platform_default(platform)
            else:
                # Standard field - use current value or default
                current_value = getattr(self, field_name, PydanticUndefined)
                if current_value is not PydanticUndefined:
                    value = current_value
                elif field_info.default is not PydanticUndefined:
                    value = field_info.default
                elif field_info.default_factory is not None:
                    value = field_info.default_factory()
                else:
                    value = PydanticUndefined

            # Serialize the value (handle Pydantic models, lists, dicts, etc.)
            if value is not PydanticUndefined:
                serialized_value = self._serl_value(value, include_none=True)
                result[field_name] = serialized_value
            else:
                result[field_name] = value

        return result

    def _clean_unset_values(self, data: t.Any) -> t.Any:
        """Recursively remove Unset values, PydanticUndefined, and empty
        dictionaries.

        Args:
            data: Data structure to clean.

        Returns:
            Cleaned data structure, or None if all values were Unset.
        """
        # Check for Unset or PydanticUndefined
        if isinstance(data, Unset) or data is PydanticUndefined:
            return None

        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Skip Unset and PydanticUndefined
                if isinstance(value, Unset) or value is PydanticUndefined:
                    continue

                cleaned_value = self._clean_unset_values(value)

                # Skip None values from recursive cleaning
                if cleaned_value is None:
                    continue

                # Skip empty dicts (all fields were Unset)
                if isinstance(cleaned_value, dict) and not cleaned_value:
                    continue

                cleaned[key] = cleaned_value

            return cleaned if cleaned else None

        if isinstance(data, list):
            cleaned_list = []
            for item in data:
                if isinstance(item, Unset) or item is PydanticUndefined:
                    continue

                cleaned_item = self._clean_unset_values(item)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)

            return cleaned_list if cleaned_list else None

        return data

    def _write_system_yaml_header(self, f: t.TextIO, platform: str) -> None:
        """Write system YAML file header.

        Args:
            f: File object.
            platform: Target platform.
        """
        import visflow

        f.write("# Visflow System Configuration\n")
        f.write(f"# Platform: {platform}\n")
        f.write(f"# Version: {visflow.__version__}\n")
        f.write("#\n")
        f.write("# This file is generated automatically. Do not edit manually.\n")
        f.write("# User configuration should be placed in ~/.config/audex/config.yml\n")
        f.write("#\n")
        f.write("# For configuration examples, see: /etc/audex/config.example.yml\n")
        f.write("\n")

    def _write_yaml_dict(
        self,
        f: t.TextIO,
        obj: dict[str, t.Any],
        indent: int = 0,
    ) -> None:
        """Write a dictionary as YAML without comments.

        Args:
            f: File object.
            obj: Dictionary to write.
            indent: Indentation level.
        """
        for key, value in obj.items():
            self._write_yaml_value(f, key, value, indent)

    def _write_yaml_value(
        self,
        f: t.TextIO,
        key: str,
        value: t.Any,
        indent: int = 0,
    ) -> None:
        """Write a single YAML key-value pair without comments.

        Args:
            f: File object.
            key: Key name.
            value: Value to write.
            indent: Indentation level.
        """
        indent_str = "  " * indent

        if value is None:
            f.write(f"{indent_str}{key}: ~\n")
        elif isinstance(value, dict):
            f.write(f"{indent_str}{key}:\n")
            self._write_yaml_dict(f, value, indent + 1)
        elif isinstance(value, list):
            f.write(f"{indent_str}{key}:\n")
            self._write_yaml_list(f, value, indent + 1)
        elif isinstance(value, str) and ("\n" in value or len(value) > 80):
            f.write(f"{indent_str}{key}: |-\n")
            for line in value.split("\n"):
                f.write(f"{indent_str}  {line}\n")
        else:
            yaml_value = self._yaml_repr(value)
            f.write(f"{indent_str}{key}: {yaml_value}\n")

    def _write_yaml_list(
        self,
        f: t.TextIO,
        items: list[t.Any],
        indent: int = 0,
    ) -> None:
        """Write a list as YAML without comments.

        Args:
            f: File object.
            items: List items to write.
            indent: Indentation level.
        """
        indent_str = "  " * indent
        for item in items:
            if isinstance(item, dict):
                items_list = list(item.items())
                if items_list:
                    first_key, first_value = items_list[0]

                    if first_value is None:
                        f.write(f"{indent_str}- {first_key}: ~\n")
                    elif isinstance(first_value, dict):
                        f.write(f"{indent_str}- {first_key}:\n")
                        self._write_yaml_dict(f, first_value, indent + 1)
                    elif isinstance(first_value, list):
                        f.write(f"{indent_str}- {first_key}:\n")
                        self._write_yaml_list(f, first_value, indent + 1)
                    elif isinstance(first_value, str) and ("\n" in first_value or len(first_value) > 80):
                        f.write(f"{indent_str}- {first_key}: |-\n")
                        for line in first_value.split("\n"):
                            f.write(f"{indent_str}    {line}\n")
                    else:
                        yaml_val = self._yaml_repr(first_value)
                        f.write(f"{indent_str}- {first_key}: {yaml_val}\n")

                    for sub_key, sub_value in items_list[1:]:
                        self._write_yaml_value(f, sub_key, sub_value, indent + 1)
            elif isinstance(item, list):
                f.write(f"{indent_str}-\n")
                self._write_yaml_list(f, item, indent + 1)
            else:
                yaml_item = self._yaml_repr(item)
                f.write(f"{indent_str}- {yaml_item}\n")

    # ============================================================================
    # Private Methods - YAML Formatting
    # ============================================================================

    def _yaml_repr(self, value: t.Any) -> str:
        """Convert a Python value to YAML representation string.

        Args:
            value: The value to convert.

        Returns:
            YAML string representation.
        """
        if value is None:
            return "~"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int | float)):
            return str(value)
        if isinstance(value, str):
            if (
                not value
                or value[0] in "-?  :,[]{}#&*!  |>'\"%@`"
                or value in ("true", "false", "null", "yes", "no", "on", "off")
                or ":" in value
                or "#" in value
            ):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"{escaped}"'
            return value

        if isinstance(value, (list | dict)):
            json_str = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            escaped = json_str.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, os.PathLike):
            try:
                return os.fspath(value)
            except Exception:
                return '""'
        if isinstance(value, PydBaseModel):
            return self._yaml_repr(json.loads(value.model_dump_json()))

        return str(value)

    # ============================================================================
    # Private Methods - Dotenv Formatting
    # ============================================================================

    def _format_env_value(self, value: t.Any) -> str:
        """Format a value for .env file with proper quoting.

        Args:
            value: The value to format.

        Returns:
            Formatted string value.
        """
        if value is None:
            return ""

        if isinstance(value, os.PathLike):
            try:
                return os.fspath(value)
            except Exception:
                return ""

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int | float)):
            return str(value)

        if isinstance(value, str):
            needs_quotes = (
                " " in value
                or "#" in value
                or "=" in value
                or "\n" in value
                or "\t" in value
                or "\r" in value
                or value.startswith(('"', "'"))
                or not value
            )

            if needs_quotes:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"{escaped}"'
            return value

        if isinstance(value, (list | dict)):
            json_str = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            escaped = json_str.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'

        str_value = str(value)
        if " " in str_value or "#" in str_value:
            escaped = str_value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return str_value

    def get_config_schema(
        self,
        *,
        mode: t.Literal["validation", "serialization"] = "validation",
        include_metadata: bool = True,
        include_examples: bool = True,
    ) -> dict[str, t.Any]:
        """Get JSON Schema for the configuration with enhanced metadata.

        Args:
            mode: Schema generation mode ("validation" or "serialization").
            include_metadata: If True, include custom metadata (constraints, examples).
            include_examples: If True, include example values in the schema.

        Returns:
            JSON Schema dictionary suitable for UI generation.
        """
        # Get base schema from Pydantic
        base_schema = self.model_json_schema(mode=mode)

        if not include_metadata:
            return base_schema

        # Enhance schema with field metadata
        field_metadata = self._collect_field_metadata()
        self._enhance_schema_with_metadata(
            base_schema,
            field_metadata,
            include_examples=include_examples,
        )

        return base_schema

    def _enhance_schema_with_metadata(
        self,
        schema: dict[str, t.Any],
        field_metadata: dict[str, dict[str, t.Any]],
        include_examples: bool = True,
        prefix: str = "",
    ) -> None:
        """Recursively enhance JSON Schema with field metadata.

        Args:
            schema: JSON Schema dictionary to enhance.
            field_metadata: Field metadata mapping.
            include_examples: Whether to include examples.
            prefix: Field path prefix for nested objects.
        """
        # Handle properties (object type)
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                field_path = f"{prefix}.{prop_name}" if prefix else prop_name
                meta = field_metadata.get(field_path, {})

                # Add constraints as schema properties
                constraints = meta.get("constraints", {})
                if "ge" in constraints:
                    prop_schema["minimum"] = constraints["ge"]
                if "gt" in constraints:
                    prop_schema["exclusiveMinimum"] = constraints["gt"]
                if "le" in constraints:
                    prop_schema["maximum"] = constraints["le"]
                if "lt" in constraints:
                    prop_schema["exclusiveMaximum"] = constraints["lt"]
                if "min_length" in constraints:
                    prop_schema["minLength"] = constraints["min_length"]
                if "max_length" in constraints:
                    prop_schema["maxLength"] = constraints["max_length"]
                if "pattern" in constraints:
                    prop_schema["pattern"] = constraints["pattern"]

                # Add examples
                if include_examples:
                    examples = meta.get("examples")
                    if examples and isinstance(examples, list) and len(examples) > 0:
                        # Extract example values
                        example_values = []
                        for ex in examples:
                            if hasattr(ex, "example"):
                                example_values.append(ex.example)
                            else:
                                example_values.append(ex)

                        # Set default to first example
                        if example_values:
                            prop_schema["default"] = example_values[0]
                            # Add all examples
                            if len(example_values) > 1:
                                prop_schema["examples"] = example_values

                # Recurse into nested objects
                if prop_schema.get("type") == "object":
                    self._enhance_schema_with_metadata(
                        prop_schema,
                        field_metadata,
                        include_examples=include_examples,
                        prefix=field_path,
                    )
                # Recurse into arrays of objects
                elif prop_schema.get("type") == "array" and "items" in prop_schema:
                    items_schema = prop_schema["items"]
                    if items_schema.get("type") == "object":
                        self._enhance_schema_with_metadata(
                            items_schema,
                            field_metadata,
                            include_examples=include_examples,
                            prefix=field_path,
                        )

        # Handle definitions/defs (nested model definitions)
        for def_key in ["definitions", "$defs"]:
            if def_key in schema:
                for _, def_schema in schema[def_key].items():
                    if def_schema.get("type") == "object":
                        self._enhance_schema_with_metadata(
                            def_schema,
                            field_metadata,
                            include_examples=include_examples,
                            prefix="",
                        )

    def ensure_no_unset(self) -> None:
        """Ensure that no fields in the settings are Unset.

        Raises:
            ValueError: If any field is Unset.
        """
        from visflow.utils import UNSET
        from visflow.utils import Unset

        def check(value: object, path: str) -> None:
            # If value is Unset, raise error
            if value is UNSET or isinstance(value, Unset):
                raise InvalidConfigurationError(f"Field '{path}' is UNSET")

            # Recurse into nested models, lists, and dicts
            if isinstance(value, PydBaseModel):
                for k, v in value:
                    check(v, f"{path}.{k}")
                return
            if isinstance(value, list):
                for i, item in enumerate(value):
                    check(item, f"{path}[{i}]")
                return
            if isinstance(value, dict):
                for k, item in value.items():
                    check(item, f"{path}.{k}")
                return

        for key, value in self:
            check(value, key)

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{field_name}={getattr(self, field_name)!r}"
            for field_name in self.__class__.model_fields  # type: ignore
        )
        return f"[Setting] {self.__class__.__name__}({field_reprs})"


class TypeBase(BaseModel):
    type: str = Field(
        ...,
        description="Discriminant for the type. Used for pydantic model discrimination.",
    )


class EnabledBase(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Whether this feature is enabled.",
    )
