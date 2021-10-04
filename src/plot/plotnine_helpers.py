"""Common functions to help with 'plotnine'."""

from enum import Enum
from typing import Union


class PlotnineUnits(str, Enum):
    """Units used in Plotnine."""

    pt = "pt"
    lines = "lines"
    inches = "in"


def margin(
    t: float = 0,
    b: float = 0,
    l: float = 0,  # noqa: E741
    r: float = 0,
    units: PlotnineUnits = PlotnineUnits.pt,
) -> dict[str, Union[float, str]]:
    """Return a dictionary of margin data.

    Args:
        t (float, optional): Top marginn. Defaults to 0.
        b (float, optional): Bottom margin. Defaults to 0.
        l (float, optional): Left margin. Defaults to 0.
        r (float, optional): Right margin. Defaults to 0.
        units (str, optional): Units for the margin. Defaults to "pt".

    Returns:
        Dict[str, Any]: A dictionary for use as margin data in plotnine plots.
    """
    if isinstance(units, str):
        units = PlotnineUnits(units)
    return {"t": t, "b": b, "l": l, "r": r, "units": units}
