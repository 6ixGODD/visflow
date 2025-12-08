from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema


class Unset:
    """A singleton class representing an unset value, distinct from None.

    This class is used to indicate that a value has not been set or
    provided, allowing differentiation between an explicit None value
    and an unset state.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unset)

    def __hash__(self) -> int:
        return hash("UNSET")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source: type[BaseModel],
        _handler: GetCoreSchemaHandler,
        /,
    ) -> CoreSchema:
        from pydantic_core.core_schema import no_info_plain_validator_function
        from pydantic_core.core_schema import plain_serializer_function_ser_schema

        return no_info_plain_validator_function(
            lambda v: v if isinstance(v, Unset) else UNSET,
            serialization=plain_serializer_function_ser_schema(lambda v: str(v)),
        )


UNSET = Unset()


def flatten_dict(
    m: t.Mapping[str, t.Any],
    /,
    sep: str = ".",
    _parent: str = "",
) -> dict[str, t.Any]:
    """Flatten a nested dictionary into a single-level dictionary with dot-
    separated keys.

    Args:
        m: The nested dictionary to flatten.
        sep: The separator to use between keys (default: '.').
        _parent: The parent key prefix (used for recursion).

    Returns:
        A flattened dictionary with dot-separated keys.
    """
    items = []  # type: list[tuple[str, t.Any]]
    for k, v in m.items():
        key = f"{_parent}{sep}{k}" if _parent else k
        if isinstance(v, t.Mapping):
            items.extend(flatten_dict(v, _parent=key, sep=sep).items())
        else:
            items.append((key, v))
    return dict(items)
