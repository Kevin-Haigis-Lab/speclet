# Common string handling functions.

import textwrap


def str_wrap(strings, width=80):
    """
    Wrap long strings to multiple lines. (vectorized)

    Parameters
    ----------
    strings: [str]
        An list (or other iterable) of strings.
    width: int
        Maximum width.

    Returns
    -------
    [str]
    """
    return ["\n".join(textwrap.wrap(x, width=width)) for x in strings]


def str_replace(strings, pattern, replace_with=" "):
    """
    Replace patterns in strings. (vectorized)

    Parameters
    ----------
    strings: [str]
        An list (or other iterable) of strings.
    pattern: str
        Pattern to be replaced.
    replace_with: str
        String to replace the pattern.

    Returns
    -------
    [str]
    """
    return [s.replace(pattern, replace_with) for s in strings]
