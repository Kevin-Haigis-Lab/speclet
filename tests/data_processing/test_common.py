#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from src.data_processing import common as dphelp

#### ---- nunique ---- ####


class TestNunique:
    def test_nunique_string(self):
        assert dphelp.nunique("abc") == 1

    def test_nunique_list_int(self):
        assert dphelp.nunique([1, 3, 2, 1]) == 3

    def test_nunique_tuple(self):
        assert dphelp.nunique((1, 2, 3, 1, 3)) == 3

    def test_nunique_dict(self):
        with pytest.raises(ValueError):
            d = {"a": 1, "b": 2, "c": 3}
            dphelp.nunique(d)


#### ---- nmutations_to_binary_array ---- ####


class TestNMutationsToBinaryArray:
    def test_nmutations_to_binary_array(self):
        np.testing.assert_equal(
            dphelp.nmutations_to_binary_array(pd.Series([0, 1, 3, 1, 0])),
            np.array([0, 1, 1, 1, 0]),
        )

    def test_empty_series_with_many_datatypes(self):
        for dtype in [int, float, "object"]:
            np.testing.assert_equal(
                dphelp.nmutations_to_binary_array(pd.Series([], dtype=dtype)),
                np.array([]),
            )


#### ---- extract_flat_ary ---- ####


def test_extract_flat_ary():
    with pytest.warns(UserWarning):
        np.testing.assert_equal(
            dphelp.extract_flat_ary(pd.Series([1, 2, 3])), np.array([1, 2, 3])
        )
    with pytest.warns(UserWarning):
        np.testing.assert_equal(
            dphelp.extract_flat_ary(pd.Series([], dtype=int)), np.array([], dtype=int)
        )


#### ---- DataFrame Helpers ---- ####


class TestIndexingFunctions:
    @pytest.fixture
    def data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": pd.Series(["a", "b", "c", "a", "c"], dtype="category"),
                "b": pd.Series([1, 2, 3, 1, 3], dtype="category"),
                "c": [1, 2, 3, 1, 3],
                "d": [3, 2, 1, 3, 2],
            }
        )

    def test_get_indices(self, data: pd.DataFrame):
        assert data["a"].dtype == "category" and data["b"].dtype == "category"
        expected_order = [0, 1, 2, 0, 2]
        np.testing.assert_equal(dphelp.get_indices(data, "a"), expected_order)
        np.testing.assert_equal(dphelp.get_indices(data, "b"), expected_order)

    def test_getting_index_on_not_cat(self, data: pd.DataFrame):
        assert data["c"].dtype == np.dtype("int64")
        with pytest.raises(AttributeError):
            _ = dphelp.get_indices(data, "c")

    def test_index_and_count(self, data: pd.DataFrame):
        idx, ct = dphelp.get_indices_and_count(data, "a")
        np.testing.assert_equal(idx, [0, 1, 2, 0, 2])
        assert ct == 3

    def test_making_categorical(self, data: pd.DataFrame):
        assert data["c"].dtype == np.dtype("int64")
        assert dphelp.make_cat(data, "c")["c"].dtype == "category"

    def test_make_cat_with_specific_order(self, data: pd.DataFrame):
        assert data["d"].dtype == np.dtype("int64")
        data = dphelp.make_cat(data.copy(), "d", ordered=True, sort_cats=False)
        np.testing.assert_equal(dphelp.get_indices(data, "d"), [0, 1, 2, 0, 1])


#### ---- center_column_grouped_dataframe ---- ####


@st.composite
def grouped_dataframe(draw) -> pd.DataFrame:
    groups = draw(st.lists(st.text(), min_size=1))
    values = [
        draw(
            st.lists(
                st.floats(
                    min_value=-1000.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
            )
        )
        for _ in groups
    ]
    return (
        pd.DataFrame({"group": groups, "value": values})
        .explode("value")
        .astype({"value": float})
        .reset_index(drop=True)
    )


@given(grouped_dataframe())
def test_center_column_grouped_dataframe(df: pd.DataFrame):
    centered_df = dphelp.center_column_grouped_dataframe(
        df, grp_col="group", val_col="value", new_col_name="centered_value"
    )
    note(centered_df)
    for g in df["group"].unique():
        vals = centered_df[centered_df["group"] == g]["centered_value"].values
        assert np.mean(vals) == pytest.approx(0.0, abs=0.001)
