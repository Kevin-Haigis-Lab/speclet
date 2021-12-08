"""Common functions to help with 'plotnine'."""

from enum import Enum
from typing import Union

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
    units: Union[str, PlotnineUnits] = PlotnineUnits.PT,
) -> dict[str, Union[float, str]]:
    """Return a dictionary of margin data.

    Args:
        t (float, optional): Top marginn. Defaults to 0.
        b (float, optional): Bottom margin. Defaults to 0.
        l (float, optional): Left margin. Defaults to 0.
        r (float, optional): Right margin. Defaults to 0.
        units (Union[str, PlotnineUnits], optional): Units for the margin. Defaults to
          "PT".

    Returns:
        Dict[str, Any]: A dictionary for use as margin data in plotnine plots.
    """
    if isinstance(units, str):
        units = PlotnineUnits(units)
    return {"t": t, "b": b, "l": l, "r": r, "units": units.value}


def set_gg_theme(figure_size: tuple[float, float] = (4, 4)) -> None:
    """Set the default plotnine theme.

    Args:
        figure_size (tuple[float, float], optional): Figure size. Defaults to (4, 4).
    """
    gg.theme_set(
        gg.theme_classic()
        + gg.theme(
            figure_size=figure_size,
            axis_ticks_major=gg.element_blank(),
            strip_background=gg.element_blank(),
        )
    )
