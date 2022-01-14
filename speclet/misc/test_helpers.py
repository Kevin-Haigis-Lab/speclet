"""Various functions used in many test modules."""

from typing import Any

#### ---- General ---- ####


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Take any arguments and do nothing.

    Returns:
        None: None
    """
    return None


#### ---- Comparisons ---- ####


def assert_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    """Compare dictionaries.

    Compares the two dictionaries using the keys from `d1` only.

    Args:
        d1 (dict[str, Any]): Dictionary one.
        d2 (dict[str, Any]): Dictionary two.
    """
    for k, v in d1.items():
        if isinstance(v, dict):
            assert_dicts(v, d2.get(k, {}))
        else:
            assert v == d2[k]
    return None
