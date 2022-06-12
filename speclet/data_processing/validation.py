"""Custom data validation checks and processes."""

from typing import Any

import numpy as np
import pandas as pd
from pandera import Check


def check_single_unique_value() -> Check:
    """Pandera check that there is only a single unique value."""
    return Check(lambda s: s.nunique() == 1)


def check_nonnegative(nullable: bool = False) -> Check:
    """Pandera check that all values are non-negative."""
    if nullable:
        return Check(lambda s: s[~np.isnan(s)] >= 0)
    else:
        return Check(lambda s: s >= 0)


def check_finite(nullable: bool = False) -> Check:
    """Pandera check that all values are finite."""
    if nullable:
        return Check(lambda x: all(np.isfinite(x[~np.isnan(x)])))
    else:
        return Check(lambda x: all(np.isfinite(x)))


def check_between(a: float, b: float) -> Check:
    """Pandera check that all values are between `a ≤ x ≤ b`."""
    return Check(lambda x: (a <= x) * (x <= b))


def check_unique_groups(grps: dict[Any, pd.Series]) -> bool | pd.Series:
    """Check that all values in a group are the same.

    This pandera check should be used with the `groupby=` option.

    Args:
        grps (dict[Any, pd.Series]): Dictionary mapping members of the grouping column
        with the data in the column being checked.

    Returns:
        Union[bool, pd.Series]: Either a single or series of results.
    """
    for data in grps.values():
        if len(data.unique()) > 1:
            return False
    return True
