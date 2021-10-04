#!/usr/bin/env python3

import pytest

from src.plot import plotnine_helpers


def test_margin_format() -> None:
    m = plotnine_helpers.margin()
    for key in ["t", "l", "b", "r", "units"]:
        assert key in m.keys()


def test_margin_values() -> None:
    values = [1, 2, 3, 4, "lines"]
    m = plotnine_helpers.margin(*values)
    for key, val in zip(["t", "b", "l", "r", "units"], values):
        assert m[key] == val


def test_margin_unit_enum() -> None:
    values = [1, 2, 3, 4, plotnine_helpers.PlotnineUnits.inches]
    m = plotnine_helpers.margin(*values)
    for key, val in zip(["t", "b", "l", "r", "units"], values):
        assert m[key] == val


def test_error_for_bad_unit() -> None:
    with pytest.raises(Exception):
        _ = plotnine_helpers.margin(units="bad-unit")
