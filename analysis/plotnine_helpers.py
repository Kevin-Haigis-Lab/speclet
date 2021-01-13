# Common functions to help with 'plotnine'.


def margin(t=0, b=0, l=0, r=0, units="pt"):
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
