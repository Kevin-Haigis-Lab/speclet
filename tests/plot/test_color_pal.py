import pytest
from matplotlib.lines import Line2D

from speclet.plot.color_pal import ColorPalette, pal_to_legend_handles


@pytest.mark.parametrize(
    "palette",
    ({"a": "blue", "b": "green"}, {1: "tomato", 2: "salmon"}),
)
def test_pal_to_legend_handles(palette: ColorPalette) -> None:
    leg = pal_to_legend_handles(palette)
    assert isinstance(leg, list)
    assert all([isinstance(i, Line2D) for i in leg])
    assert len(leg) == len(palette)
