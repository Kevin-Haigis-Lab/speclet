from enum import EnumMeta

import pytest

from speclet.plot import color_pal


@pytest.mark.parametrize(
    "palette",
    (color_pal.SeabornColor, color_pal.ModelColors, color_pal.FitMethodColors),
)
def test_make_pal(palette: EnumMeta) -> None:
    pal_dict = color_pal.make_pal(palette)
    assert isinstance(pal_dict, dict)
    assert all([isinstance(k, str) for k in pal_dict.keys()])
    assert all([isinstance(v, str) for v in pal_dict.values()])
    assert len(pal_dict) > 0
