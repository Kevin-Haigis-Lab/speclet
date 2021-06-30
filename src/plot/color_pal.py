#!/usr/bin/env python3

"""My colors and color palettes."""

from enum import Enum, EnumMeta
from typing import Dict


class SeabornColor(str, Enum):
    """Colors from the 'seaborn' package."""

    ORANGE = "#ED854A"
    BLUE = "#4878D0"
    GREEN = "#6BCC64"
    RED = "#D65F5F"


class ModelColors(str, Enum):
    """Colors for specific models."""

    CERES = "#417BB1"
    CERES_MIMIC = "#853FA2"
    SPECLET_ONE = "#009C77"
    SPECLET_TWO = "#DE522F"


class FitMethodColors(str, Enum):
    """Colors for the different ways to fit a model."""

    PYMC3_MCMC = "#F57E3F"
    PYMC3_ADVI = "#0AA8A2"


def make_pal(colors: EnumMeta) -> Dict[str, str]:
    """Convert an Enum to a usable color palette.

    Args:
        colors (EnumMeta): Enum to convert to a palette.

    Returns:
        (Dict[str, str]) Dictionary mapping of enum.
    """
    d: Dict[str, str] = {}
    for c in colors:  # type: ignore
        assert isinstance(c.name, str) and isinstance(c.value, str)
        d[c.name] = c.value
    return d
