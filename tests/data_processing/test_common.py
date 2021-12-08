from typing import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from speclet.data_processing import common as dphelp

#### ---- nunique ---- ####


class TestNunique:
    def test_nunique_string(self) -> None:
        assert dphelp.nunique("abc") == 1

    def test_nunique_list_int(self) -> None:
        assert dphelp.nunique([1, 3, 2, 1]) == 3

    def test_nunique_tuple(self) -> None:
        assert dphelp.nunique((1, 2, 3, 1, 3)) == 3

    def test_nunique_dict(self) -> None:
        with pytest.raises(ValueError):
            d = {"a": 1, "b": 2, "c": 3}
            dphelp.nunique(d)


#### ---- nmutations_to_binary_array ---- ####


class TestNMutationsToBinaryArray:
    def test_nmutations_to_binary_array(self) -> None:
        np.testing.assert_equal(
            dphelp.nmutations_to_binary_array(pd.Series([0, 1, 3, 1, 0])),
            np.array([0, 1, 1, 1, 0]),
        )

    def test_empty_series_with_many_datatypes(self) -> None:
        for dtype in [int, float, "object"]:
            np.testing.assert_equal(
                dphelp.nmutations_to_binary_array(pd.Series([], dtype=dtype)),
                np.array([]),
            )


#### ---- extract_flat_ary ---- ####


def test_extract_flat_ary() -> None:
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

    def test_get_indices(self, data: pd.DataFrame) -> None:
        assert data["a"].dtype == "category" and data["b"].dtype == "category"
        expected_order = [0, 1, 2, 0, 2]
        np.testing.assert_equal(dphelp.get_indices(data, "a"), expected_order)
        np.testing.assert_equal(dphelp.get_indices(data, "b"), expected_order)

    def test_getting_index_on_not_cat(self, data: pd.DataFrame) -> None:
        assert data["c"].dtype == np.dtype("int64")
        with pytest.raises(AttributeError):
            _ = dphelp.get_indices(data, "c")

    def test_index_and_count(self, data: pd.DataFrame) -> None:
        idx, ct = dphelp.get_indices_and_count(data, "a")
        np.testing.assert_equal(idx, [0, 1, 2, 0, 2])
        assert ct == 3

    def test_making_categorical(self, data: pd.DataFrame) -> None:
        assert data["c"].dtype == np.dtype("int64")
        assert dphelp.make_cat(data, "c")["c"].dtype == "category"

    def test_make_cat_with_specific_order(self, data: pd.DataFrame) -> None:
        assert data["d"].dtype == np.dtype("int64")
        data = dphelp.make_cat(data.copy(), "d", ordered=True, sort_cats=False)
        np.testing.assert_equal(dphelp.get_indices(data, "d"), [2, 1, 0, 2, 1])


#### ---- center_column_grouped_dataframe ---- ####


@st.composite
def grouped_dataframe(draw: Callable) -> pd.DataFrame:
    print(draw)
    groups = draw(st.lists(st.text(), min_size=1))
    groups = [g.encode("ascii", "ignore") for g in groups]
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
def test_center_column_grouped_dataframe(df: pd.DataFrame) -> None:
    centered_df = dphelp.center_column_grouped_dataframe(
        df, grp_col="group", val_col="value", new_col_name="centered_value"
    )
    note(centered_df)
    for g in df["group"].unique():
        vals = centered_df[centered_df["group"] == g]["centered_value"].values
        assert np.mean(vals) == pytest.approx(0.0, abs=0.001)


#### ---- dataframe_to_matrix ---- ####


def mock_df_for_df_to_mat(row_n: int, col_n: int) -> pd.DataFrame:
    a = [f"row_{i}" for i in range(row_n)]
    b = [f"col_{i}" for i in range(col_n)]
    a_col = np.repeat(a, col_n)
    b_col = np.tile(b, row_n)
    df = pd.DataFrame(
        {
            "A": a_col,
            "B": b_col,
            "vals": np.random.randint(0, 10, len(a_col)),
            "faker": "hi",
        }
    )
    assert df[["A", "B"]].drop_duplicates().shape[0] == df.shape[0]
    assert df.shape == (row_n * col_n, 4)
    return df


@given(st.integers(1, 6), st.integers(1, 6))
def test_dataframe_to_matrix(row_n: int, col_n: int) -> None:
    df = mock_df_for_df_to_mat(row_n, col_n)
    m = dphelp.dataframe_to_matrix(df, rows="A", cols="B", values="vals")

    assert isinstance(m, np.ndarray)
    assert m.shape == (row_n, col_n)
    for i in range(row_n):
        for j in range(col_n):
            df_val = (
                df.query(f"A == 'row_{i}'").query(f"B == 'col_{j}'")["vals"].values[0]
            )
            m_val = m[i, j]
            assert df_val == m_val


@given(st.integers(1, 6), st.integers(1, 6))
def test_dataframe_to_matrix_categorical(row_n: int, col_n: int) -> None:
    df = mock_df_for_df_to_mat(row_n, col_n)
    df["A"] = pd.Categorical(df["A"].values, categories=df["A"].unique(), ordered=True)
    df["B"] = pd.Categorical(df["B"].values, categories=df["B"].unique(), ordered=True)
    df = df.sample(frac=1.0).reset_index(drop=True)
    m = dphelp.dataframe_to_matrix(df, rows="A", cols="B", values="vals")

    assert isinstance(m, np.ndarray)
    assert m.shape == (row_n, col_n)
    for i in range(row_n):
        for j in range(col_n):
            df_val = (
                df.query(f"A == 'row_{i}'").query(f"B == 'col_{j}'")["vals"].values[0]
            )
            m_val = m[i, j]
            assert df_val == m_val
