# Common functions to help with 'plotnine'.

from typing import Any, Dict, List

PLOTNINE_UNITS = ["pt", "lines", "in"]


def get_possible_units() -> List[str]:
    return PLOTNINE_UNITS


def margin(
    t: float = 0, b: float = 0, l: float = 0, r: float = 0, units: str = "pt"
) -> Dict[str, Any]:
    """
    Return a dictionary of margin data.
    """

    if not units in PLOTNINE_UNITS:
        raise ValueError(f"Unit of type {units} is not available.")

    return {"t": t, "b": b, "l": l, "r": r, "units": units}
