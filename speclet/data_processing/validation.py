"""Custom data validation checks and processes."""

import numpy as np
from pandera import Check


def check_positive() -> Check:
    return Check(lambda s: s > 0)


def check_finite() -> Check:
    return Check(lambda x: all(np.isfinite(x)))
