"""My colors and color palettes."""

from enum import Enum, EnumMeta


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


def make_pal(colors: EnumMeta) -> dict[str, str]:
    """Convert an Enum to a usable color palette.

    Args:
        colors (EnumMeta): Enum to convert to a palette.

    Returns:
        (Dict[str, str]) Dictionary mapping of enum.
    """
    d: dict[str, str] = {}
    for c in colors:  # type: ignore
        assert isinstance(c.name, str) and isinstance(c.value, str)
        d[c.name] = c.value
    return d
