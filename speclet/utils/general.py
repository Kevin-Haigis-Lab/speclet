"""General utilities."""

from typing import Iterable, TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")


def resolve_optional_kwargs(d: dict[_T, _U] | None) -> dict[_T, _U]:
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


def merge_sets(sets: Iterable[set[_T]]) -> set[_T]:
    """Set a collection of sets.

    Args:
        sets (Iterable[set[_T]]): Collections of sets.

    Returns:
        set[_T]: Merged set.
    """
    new_set: set[_T] = set()
    for sub_set in sets:
        new_set = new_set.union(sub_set)
    return new_set
