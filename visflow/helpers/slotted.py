from __future__ import annotations

import typing as t


class FieldInfo:
    __slots__ = ("default", "default_factory", "required")

    def __init__(
        self,
        default: t.Any | ellipsis = Ellipsis,
        default_factory: t.Callable[[], t.Any] | None = None,
        required: bool = True,
    ):
        self.default = default
        self.default_factory = default_factory
        self.required = required

    def get_default(self) -> t.Any:
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


class SlottedDataClass:
    __slots__ = ()  # type: t.Collection[str]

    _field_defaults: t.ClassVar[t.Dict[str, t.Any]] = {}
    _field_optional: t.ClassVar[t.Set[str]] = set()
    _field_info: t.ClassVar[t.Dict[str, FieldInfo]]
    _all_fields: t.ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: t.Any):
        super().__init_subclass__(**kwargs)
        cls._field_info = cls._parse_field()
        cls._all_fields = frozenset[str](
            name for name in cls.__slots__ if not name.startswith("_")
        )

    @classmethod
    def _parse_field(cls) -> t.Dict[str, FieldInfo]:
        field_info = {}
        anns = getattr(cls, "__annotations__", {})
        defaults = getattr(cls, "_field_defaults", {})
        optional: t.Set[str] = getattr(cls, "_field_optional", set())

        for field_name in getattr(cls, "__slots__", []):
            if field_name.startswith("_"):
                continue

            field_type = anns.get(field_name, t.Any)

            if field_name in defaults:
                default_val = defaults[field_name]
                if callable(default_val) and not isinstance(
                    default_val, (str, int, float, bool)
                ):
                    info = FieldInfo(default_factory=default_val, required=False)
                else:
                    info = FieldInfo(default=default_val, required=False)
            elif field_name in optional:
                info = FieldInfo(required=False, default=None)
            else:
                is_optional = cls._is_optional(field_type)
                info = FieldInfo(
                    required=not is_optional, default=None if is_optional else ...
                )

            field_info[field_name] = info

        return field_info

    def __init__(self, **kwargs: t.Any):
        if not hasattr(self, "__slots__"):
            raise TypeError(f"{self.__class__.__name__} must define __slots__")

        required_fields = {
            name for name, info in self._field_info.items() if info.required
        }
        missing_required = required_fields - kwargs.keys()

        if missing_required:
            raise ValueError(f"Missing required fields: {missing_required}")

        for field_name in self.__slots__:
            if field_name in kwargs:
                object.__setattr__(self, field_name, kwargs[field_name])
            elif field_name in self._field_info:
                info = self._field_info[field_name]
                if not info.required:
                    object.__setattr__(self, field_name, info.get_default())

    @staticmethod
    def _is_optional(field_t: t.Any) -> bool:
        origin = t.get_origin(field_t)
        if origin is t.Union:
            args = t.get_args(field_t)
            return len(args) == 2 and type(None) in args
        return False

    @classmethod
    def from_dict(cls: t.Type[t.Self], data: t.Dict[str, t.Any]) -> t.Self:
        return cls(**data)

    def to_dict(self, recursive: bool = False) -> t.Dict[str, t.Any]:
        if recursive:
            return self._to_dict_recursive()

        return {
            field_name: getattr(self, field_name)
            for field_name in self._all_fields
            if hasattr(self, field_name)
        }

    def _to_dict_recursive(self) -> t.Dict[str, t.Any]:
        result = {}
        for field_name in self._all_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                result[field_name] = self._serialize_value(value)
        return result

    @classmethod
    def _serialize_value(cls, value: t.Any) -> t.Any:
        if isinstance(value, SlottedDataClass):
            return value.to_dict(recursive=True)
        elif isinstance(value, (list, tuple)):
            return [cls._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: cls._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, set):
            return [cls._serialize_value(item) for item in value]
        else:
            return value

    def __repr__(self) -> str:
        parts = []
        for field_name in self._all_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                parts.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(parts)})"
