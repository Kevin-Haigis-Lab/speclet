#!/usr/bin/env python3

"""My colors and color palettes."""

from enum import Enum


class SeabornColor(str, Enum):
    """Colors from the 'seaborn' package."""

    orange = "#ED854A"
    blue = "#4878D0"
    green = "#6BCC64"
    red = "#D65F5F"


class ModelColors(str, Enum):
    """Colors for specific models."""

    CERES = "#417BB1"
    CERES_mimic = "#853FA2"
    speclet_one = "#009C77"
