from typing import List

import arviz as az
import numpy as np
import pandas as pd
import pytest

from src.pipelines import collate_sbc_cli as csbc


@pytest.fixture
def centered_eight() -> az.InferenceData:
    x = az.load_arviz_data("centered_eight")
    assert isinstance(x, az.InferenceData)
    return x


@pytest.fixture
def centered_eight_post(centered_eight: az.InferenceData) -> pd.DataFrame:
    x = az.summary(centered_eight)
    assert isinstance(x, pd.DataFrame)
    return x


def test_get_hdi_colnames_from_az_summary(centered_eight_post: pd.DataFrame) -> None:
    hdi_cols = csbc._get_hdi_colnames_from_az_summary(centered_eight_post)
    assert hdi_cols == ("hdi_3%", "hdi_97%")


def test_is_true_value_within_hdi_lower_limit() -> None:
    n = 100
    low = pd.Series(list(range(0, n)))
    high = pd.Series([200] * n)
    vals = pd.Series([50] * n)
    is_within = csbc._is_true_value_within_hdi(low, vals, high)
    assert np.all(is_within[:50])
    assert not np.any(is_within[50:])


def test_is_true_value_within_hdi_upper_limit() -> None:
    n = 100
    low = pd.Series([0] * n)
    high = pd.Series(list(range(100)))
    vals = pd.Series([50] * n)
    is_within = csbc._is_true_value_within_hdi(low, vals, high)
    assert not np.any(is_within[:51])
    assert np.all(is_within[51:])


def test_get_prior_value_using_index_list_empty_idx() -> None:
    a = np.array([4, 3, 2, 1])
    idx: List[int] = []
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 4


def test_get_prior_value_using_index_list_1d() -> None:
    a = np.array([4, 3, 2, 1])
    idx = [0]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 4
    idx = [1]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 3


def test_get_prior_value_using_index_list_2d() -> None:
    a = np.arange(9).reshape((3, 3))
    print(a)
    idx = [1, 2]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == a[1, 2]


@pytest.mark.parametrize(
    "p, res",
    [
        ("a", ["a"]),
        ("abc", ["abc"]),
        ("abc[0]", ["abc", "0"]),
        ("abc[0,2,5]", ["abc", "0", "2", "5"]),
        ("abc[ x, y, z]", ["abc", " x", " y", " z"]),
        ("abc[x,y,z]", ["abc", "x", "y", "z"]),
    ],
)
def test_split_parameter(p: str, res: str) -> None:
    assert res == csbc._split_parameter(p)
