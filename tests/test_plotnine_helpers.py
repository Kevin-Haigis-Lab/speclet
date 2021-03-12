#!/usr/bin/env python3

import pytest

from src import plotnine_helpers


def test_margin_format():
    m = plotnine_helpers.margin()

    for key in ["t", "l", "b", "r", "units"]:
        assert key in m.keys()


def test_margin_values():
    values = [1, 2, 3, 4, "lines"]
    m = plotnine_helpers.margin(*values)
    for key, val in zip(["t", "b", "l", "r", "units"], values):
        assert m[key] == val


def test_error_for_bad_unit():
    with pytest.raises(ValueError):
        _ = plotnine_helpers.margin(units="bad-unit")
