"""General utilities."""

from typing import Optional, TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")


def resolve_optional_kwargs(d: Optional[dict[_T, _U]]) -> dict[_T, _U]:
    """Resolve optional dictionary to an empty dictionary instead of None.

    Args:
        d (Optional[dict[_T, _U]]): Optional dictionary.

    Returns:
        dict[_T, _U]: Either the input dictionary or a new empty dictionary.
    """
    if d is None:
        empty_d: dict[_T, _U] = {}
        return empty_d
    return d
