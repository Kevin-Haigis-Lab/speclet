"""My colors and color palettes."""

from enum import Enum
from typing import Literal


class SeabornColor(Enum):
    """Colors from the 'seaborn' package."""

    ORANGE = "#ED854A"
    BLUE = "#4878D0"
    GREEN = "#6BCC64"
    RED = "#D65F5F"


class ModelColors(Enum):
    """Colors for specific models."""

    CERES = "#417BB1"
    CERES_MIMIC = "#853FA2"
    SPECLET_ONE = "#009C77"
    SPECLET_TWO = "#DE522F"


class FitMethodColors(Enum):
    """Colors for the different ways to fit a model."""

    PYMC3_MCMC = "#F57E3F"
    PYMC3_ADVI = "#0AA8A2"


ColorPalette = Literal[SeabornColor, ModelColors, FitMethodColors]  # type: ignore
name = str
color = str


def make_pal(colors: ColorPalette) -> dict[name, color]:
    """Convert an Enum to a usable color palette.

    Args:
        colors (ColorPalette): Color palette Enum to convert to dictionary.

    Returns:
        (dict[name, color]) Dictionary mapping of name to color (both strings).
    """
    d: dict[str, str] = {}
    for c in colors:
        d[c.name] = c.value
    return d
