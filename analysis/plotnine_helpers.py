# Common functions to help with 'plotnine'.

from typing import Any, Dict


def margin(
    t: int = 0, b: int = 0, l: int = 0, r: int = 0, units: str = "pt"
) -> Dict[str, Any]:
    """
    Return a dictionary of margin data.

    Parameters
    ----------
    t, b, l, r: num
        The margins for the top, bottom, left, and right.
    units: str
        The units to use for the margin. Options are "pt", "lines", and "in".

    Returns
    -------
    dict
    """
    return {"t": t, "b": b, "l": l, "r": r, "units": units}
