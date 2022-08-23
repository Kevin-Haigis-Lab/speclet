"""My colors and color palettes."""

from typing import Any

from matplotlib.lines import Line2D

name = str
color = str
ColorPalette = dict[name, color]


def pal_to_legend_handles(color_pal: ColorPalette, **kwargs: Any) -> list[Line2D]:
    """Convert a palette into legend handles for matplotlib.

    Args:
        color_pal (ColorPalette): Color palette.

    Returns:
        list[Line2D]: Legend handles.
    """
    handles: list[Line2D] = []
    for lbl, color in color_pal.items():
        handles.append(Line2D([0], [0], label=lbl, color=color, **kwargs))
    return handles
