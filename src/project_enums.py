"""Project-wide enum classes."""

from enum import Enum, unique
from typing import NoReturn

#### ---- Models and modeling ---- ####


@unique
class ModelFitMethod(str, Enum):
    """Available fit methods."""

    ADVI = "ADVI"
    MCMC = "MCMC"


@unique
class ModelParameterization(Enum):
    """Possible model parameterization methods."""

    CENTERED = "centered"
    NONCENTERED = "noncentered"


@unique
class ModelOption(str, Enum):
    """Model options."""

    SPECLET_TEST_MODEL = "speclet-test-model"
    CRC_CERES_MIMIC = "crc-ceres-mimic"
    SPECLET_ONE = "speclet-one"
    SPECLET_TWO = "speclet-two"
    SPECLET_FOUR = "speclet-four"
    SPECLET_FIVE = "speclet-five"
    SPECLET_SIX = "speclet-six"
    SPECLET_SEVEN = "speclet-seven"


#### ---- SLURM and pipelines ---- ####


@unique
class SlurmPartitions(str, Enum):
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


#### ---- Exhaustiveness checks ---- ####


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhausstiveness.

    Args:
        value (NoReturn): Some value passed as an enum value.

    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"
