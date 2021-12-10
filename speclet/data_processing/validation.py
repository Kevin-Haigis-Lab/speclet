"""Custom data validation checks and processes."""

import numpy as np
from pandera import Check


def check_nonnegative() -> Check:
    """Pandera check that all values are non-negative."""
    return Check(lambda s: s >= 0)


def check_finite() -> Check:
    """Pandera check that all values are finite."""
    return Check(lambda x: all(np.isfinite(x)))


def check_between(a: float, b: float) -> Check:
    """Pandera check that all values are between `a ≤ x ≤ b`."""
    return Check(lambda x: (a <= x) * (x <= b))
