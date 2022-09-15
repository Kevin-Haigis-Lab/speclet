"""Plotting."""

from enum import Enum

import matplotlib as mpl
import matplotlib.pyplot as plt

from speclet.io import dissertation_figure_stylesheet


class PlottingMode(str, Enum):
    """Specific plotting modes."""

    DEFAULT = "DEFAULT"
    DISSERTATION = "DISSERTATION"
    PAPER = "PAPER"


def set_speclet_theme(mode: PlottingMode | str = PlottingMode.DEFAULT) -> None:
    """Set the plot theme to the 'spleclet' project theme."""
    if isinstance(mode, str):
        mode = PlottingMode(mode)

    plt.style.use(["seaborn-whitegrid"])
    if mode is PlottingMode.DISSERTATION:
        plt.style.use(dissertation_figure_stylesheet())
    return None


def align_legend_title(leg: mpl.legend.Legend, ha: str = "left") -> None:
    """Align a legend title."""
    leg._legend_box.align = ha
