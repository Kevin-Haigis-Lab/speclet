"""Common and general vector operations."""

from typing import Callable, Optional

import numpy as np
from scipy import stats


def zscale(x: np.ndarray) -> np.ndarray:
    """Z-scale an array.

    Args:
        x (np.ndarray): Numeric array.

    Returns:
        np.ndarray: Z-scaled array.
    """
    return stats.zscore(x)


def squish(x: float, lower: float, upper: float) -> float:
    """Squish an array to lie between a lower an upper bound.

    Args:
        x (float): Numeric array.
        lower (float): Lower bound (minimum value).
        upper (float): Upper bound (maximum value).

    Returns:
        float: The numeric array with no values below the lower bound and none above the
        upper bound.
    """
    return max(min(x, upper), lower)


squish_array = np.vectorize(squish)


def np_identity(x: np.ndarray) -> np.ndarray:
    """Identity function.

    Args:
        x (np.ndarray): Numpy array.

    Returns:
        np.ndarray: The exact same numpy array with no changes.
    """
    return x


NumpyTransform = Callable[[np.ndarray], np.ndarray]


def careful_zscore(
    x: np.ndarray,
    atol: float = 0.01,
    transform: Optional[NumpyTransform] = None,
) -> np.ndarray:
    """Z-scale an array carefully.

    Z-scaling can be very unstable if the values are very close together because the
    standard deviation is very small. For the case of modeling, it is logical to just
    set the values to 0 so that the covariate is canceled out. If any of the following
    critera are met, the z-score is ignored and an array of 0's is returned:

    1. If there is only one value.
    2. If the values are all close to 0.
    3. If the values are all very close together.

    Args:
        x (np.ndarray): Array to rescale.
        atol (float, optional): Absolute tolerance for variance in values to decide if
          the z-score should be applied. Defaults to 0.01.
        transform (Optional[NumpyTransform], optional): A transformation to apply to the
          array before z-scaling, but after checking the variance. Defaults to None
          which is equivalent to an identity transform.

    Returns:
        np.ndarray: Z-scaled array or array of 0's.
    """
    if transform is None:
        transform = np_identity

    if len(x) == 1:
        # If there is only one value, set z-scaled expr to 0.
        return np.zeros_like(x)
    elif np.allclose(x, 0.0, atol=atol):
        # If all values are close to 0, set value to 0.
        return np.zeros_like(x)
    elif np.allclose(x, np.mean(x), atol=atol):
        # If all values are about equal, set value to 0.
        return np.zeros_like(x)
    else:
        return zscale(transform(x))
