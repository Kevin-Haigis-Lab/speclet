"""Project-wide enum classes."""

from enum import Enum


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"
