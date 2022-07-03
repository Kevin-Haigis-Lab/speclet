"""Common string handling functions."""

import textwrap
from typing import Iterable


def str_wrap(strings: str | Iterable[str], width: int = 80) -> Iterable[str]:
    """Wrap long strings to multiple lines (vectorized).

    Args:
        strings (Union[str, Iterable[str]]): Strings to wrap.
        width (int, optional): Number of characters to wrap at. Defaults to 80.

    Returns:
        Iterable[str]: Strings wrapped to the specified length.
    """
    if isinstance(strings, str):
        return "\n".join(textwrap.wrap(strings, width=width))
    else:
        return ["\n".join(textwrap.wrap(x, width=width)) for x in strings]


def str_replace(
    strings: Iterable[str] | str, pattern: str, replace_with: str = " "
) -> Iterable[str]:
    """Replace patterns in strings (vectorized).

    Args:
        strings (Union[Iterable[str], str]): Strings to edit.
        pattern (str): The pattern to replace.
        replace_with (str, optional): The replacement text. Defaults to " ".

    Returns:
        Iterable[str]: Modified strings.
    """
    if isinstance(strings, str):
        return strings.replace(pattern, replace_with)
    else:
        return [s.replace(pattern, replace_with) for s in strings]


def prefixed_count(prefix: str, n: int, plus: float = 0) -> list[str]:
    """Make an array of 1-->n with the number and some prefix.

    Args:
        prefix (str): A prefix for each number.
        n (int): The number to count to.
        plus (float): The starting point for the count. Defaults to 0.0.

    Returns:
        Iterable[str]: Modified strings.
    """
    return [prefix + str(i + plus) for i in range(n)]
