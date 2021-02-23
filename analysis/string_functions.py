# Common string handling functions.

import textwrap
from typing import Iterable, List, Union


def str_wrap(strings: Union[str, Iterable[str]], width: int = 80) -> Iterable[str]:
    """
    Wrap long strings to multiple lines. (vectorized)
    """
    if isinstance(strings, str):
        return "\n".join(textwrap.wrap(strings, width=width))
    else:
        return ["\n".join(textwrap.wrap(x, width=width)) for x in strings]


def str_replace(
    strings: Union[Iterable[str], str], pattern: str, replace_with: str = " "
) -> Iterable[str]:
    """
    Replace patterns in strings. (vectorized)
    """
    if isinstance(strings, str):
        return strings.replace(pattern, replace_with)
    else:
        return [s.replace(pattern, replace_with) for s in strings]


def prefixed_count(prefix: str, n: int, plus: float = 0) -> List[str]:
    """
    Make an array of 1-->n with the number and some prefix.
    """
    return [prefix + str(i + plus) for i in range(n)]
