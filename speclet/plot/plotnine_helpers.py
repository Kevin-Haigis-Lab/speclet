"""Common functions to help with 'plotnine'."""

import warnings
from enum import Enum
from typing import Any

import plotnine as gg


class PlotnineUnits(str, Enum):
    """Units used in Plotnine."""

    PT = "pt"
    LINES = "lines"
    INCHES = "in"


def margin(
    t: float = 0,
    b: float = 0,
    l: float = 0,  # noqa: E741
    r: float = 0,
    units: str | PlotnineUnits = PlotnineUnits.PT,
) -> dict[str, float | str]:
    """Return a dictionary of margin data.

    Args:
        t (float, optional): Top marginn. Defaults to 0.
        b (float, optional): Bottom margin. Defaults to 0.
        l (float, optional): Left margin. Defaults to 0.
        r (float, optional): Right margin. Defaults to 0.
        units (str | PlotnineUnits, optional): Units for the margin. Defaults to
          "PT".

    Returns:
        dict[str, float | str]: A dictionary for use as margin data in plotnine plots.
    """
    if isinstance(units, str):
        units = PlotnineUnits(units)
    return {"t": t, "b": b, "l": l, "r": r, "units": units.value}


def set_gg_theme(figure_size: tuple[float, float] = (4.0, 4.0)) -> None:
    """Set the default plotnine theme.

    Args:
        figure_size (tuple[float, float], optional): Figure size. Defaults to (4, 4).
    """
    warnings.warn("Deprecated - use `set_theme_speclet()` instead.")
    set_theme_speclet()
    return None


class theme_speclet(gg.theme_bw):
    """Plotnine theme for the 'speclet' project."""

    def __init__(self, base_size: int = 10, base_family: str | None = None):
        """Plotnine theme for the 'speclet' project.

        Args:
            base_size (int, optional): Base font size. Defaults to 10.
            base_family ( str | None, optional): Font family. Defaults to None.
        """
        gg.theme_bw.__init__(self, base_size, base_family)
        self.add_theme(
            gg.theme(
                panel_border=gg.element_blank(),
                axis_ticks=gg.element_blank(),
                plot_margin=0,
                strip_background=gg.element_blank(),
                figure_size=(5, 3),
            ),
            inplace=True,
        )


def set_theme_speclet() -> None:
    """Set the 'plotnine' theme to the 'speclet' project theme."""
    gg.theme_set(theme_speclet)
    return None


class scale_color_heatmap(gg.scale_color_gradient2):
    """Heatmap color scale."""

    def __init__(self, midpoint: float = 0, **kwargs: dict[str, Any]) -> None:
        """Heatmap color gradient.

        Keyword arguments passed to `plotnine.scale_color_gradient2()`.
        """
        super().__init__(
            low="#2c7bb6", mid="#ffffbf", high="#d7191c", midpoint=midpoint, **kwargs
        )
        return None


class scale_fill_heatmap(gg.scale_fill_gradient2):
    """Heatmap fill."""

    def __init__(self, midpoint: float = 0, **kwargs: dict[str, Any]) -> None:
        """Heatmap fill gradient.

        Keyword arguments passed to `plotnine.scale_fill_gradient2()`.
        """
        super().__init__(
            low="#2c7bb6", mid="#ffffbf", high="#d7191c", midpoint=midpoint, **kwargs
        )
        return None
