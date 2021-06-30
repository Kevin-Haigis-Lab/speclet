"""Project-wide enum classes."""

from enum import Enum
from typing import NoReturn


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"


#### ---- Exhaustiveness checks ---- ####


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhausstiveness.

    Args:
        value (NoReturn): Some value passed as an enum value.

    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"
