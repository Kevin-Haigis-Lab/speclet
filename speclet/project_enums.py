"""Project-wide enum classes."""

from enum import Enum, unique
from typing import NoReturn

#### ---- Models and modeling ---- ####


@unique
class ModelFitMethod(Enum):
    """Available fit methods."""

    PYMC_MCMC = "PYMC_MCMC"
    PYMC_ADVI = "PYMC_ADVI"
    PYMC_NUMPYRO = "PYMC_NUMPYRO"


@unique
class ModelParameterization(Enum):
    """Possible model parameterization methods."""

    CENTERED = "centered"
    NONCENTERED = "noncentered"


#### ---- SLURM and pipelines ---- ####


@unique
class SlurmPartitions(Enum):
    """Partitions of the HPC available through SLURM."""

    PRIORITY = "priority"
    INTERACTIVE = "interactive"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@unique
class SpecletPipeline(Enum):
    """Pipelines available in this project."""

    FITTING = "fitting"
    SBC = "sbc"


@unique
class MockDataSize(Enum):
    """Options for dataset seizes when generating mock data."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


#### ---- Exhaustiveness checks ---- ####


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhaustiveness.

    Args:
        value (NoReturn): Some value passed as an enum value.

    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"  # noqa: B011
