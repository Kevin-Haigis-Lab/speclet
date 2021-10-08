from pathlib import Path

import pandas as pd
import pytest

from src.context_managers import dask_client
from src.managers.data_managers import CrisprScreenDataManager


def reverse_sgrna(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(rev_sgrna=lambda d: d.sgrna.values[::-1])


dask_kwargs = {"n_workers": 2, "threads_per_worker": 1, "memory_limit": "500MB"}


# ---- Reading data ----


@pytest.mark.parametrize("use_dask", (False, True))
def test_get_real_data(depmap_test_data: Path, use_dask: bool) -> None:
    dm = CrisprScreenDataManager(depmap_test_data, use_dask=use_dask)
    if use_dask:
        with dask_client(**dask_kwargs):  # type: ignore
            df = dm.get_data()
    else:
        df = dm.get_data()
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert all([c in df.columns for c in ("sgrna", "hugo_symbol", "depmap_id")])


@pytest.mark.parametrize("use_dask", (False, True))
def test_custom_transformation(depmap_test_data: Path, use_dask: bool) -> None:
    dm = CrisprScreenDataManager(depmap_test_data, transformations=[reverse_sgrna])
    if use_dask:
        with dask_client(**dask_kwargs):  # type: ignore
            df = dm.get_data()
    else:
        df = dm.get_data()
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "rev_sgrna" in df.columns
    assert all(df.sgrna.values == df.rev_sgrna.values[::-1])
    assert all([c in df.columns for c in ("sgrna", "hugo_symbol", "depmap_id")])


@pytest.mark.parametrize("use_dask", (False, True))
def test_select_columns(depmap_test_data: Path, use_dask: bool) -> None:
    keep_cols = {"sgrna", "hugo_symbol"}
    dm = CrisprScreenDataManager(depmap_test_data, columns=keep_cols.copy())
    if use_dask:
        with dask_client(**dask_kwargs):  # type: ignore
            df = dm.get_data()
    else:
        df = dm.get_data()
    assert df.shape[0] > 0 and df.shape[1] == 2
    assert all([c in df.columns for c in keep_cols])
    assert "depmap_id" not in df.columns


@pytest.mark.parametrize("use_dask", (False, True))
def test_select_columns_and_transformation(
    depmap_test_data: Path, use_dask: bool
) -> None:
    keep_cols = {"sgrna", "hugo_symbol"}
    dm = CrisprScreenDataManager(
        depmap_test_data, transformations=[reverse_sgrna], columns=keep_cols.copy()
    )
    if use_dask:
        with dask_client(**dask_kwargs):  # type: ignore
            df = dm.get_data()
    else:
        df = dm.get_data()
    assert df.shape[0] > 0 and df.shape[1] == 3
    assert all(df.sgrna.values == df.rev_sgrna.values[::-1])
    assert all([c in df.columns for c in keep_cols])
    assert "rev_sgrna" in df.columns
    assert "depmap_id" not in df.columns


def test_clear_data(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(depmap_test_data)
    _ = dm.get_data()
    assert dm.data_is_loaded()
    dm.clear_data()
    assert not dm.data_is_loaded()


# ---- Transformations ----


def double_log_fold_change(df: pd.DataFrame) -> pd.DataFrame:
    df["lfc"] = df["lfc"] * 2.0
    return df


def add_one_to_log_fold_change(df: pd.DataFrame) -> pd.DataFrame:
    df["lfc"] = df["lfc"] + 1.0
    return df


def test_add_transformation(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(depmap_test_data)
    df = dm.get_data()
    assert len(dm.get_transformations()) == 1
    dm.add_transformation(double_log_fold_change)
    assert len(dm.get_transformations()) == 2
    dm.clear_data()
    assert not dm.data_is_loaded()
    new_df = dm.get_data()
    assert all(new_df["lfc"].values == (df["lfc"].values * 2.0))


def test_add_transformations(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(depmap_test_data)
    assert len(dm.get_transformations()) == 1
    dm.add_transformation([double_log_fold_change, add_one_to_log_fold_change])
    assert len(dm.get_transformations()) == 3


def test_insert_transformation(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(
        depmap_test_data, transformations=[add_one_to_log_fold_change]
    )

    df1 = dm.get_data()  # lfc = lfc + 1

    dm.clear_data()
    dm.insert_transformation(double_log_fold_change, at=0)
    df2 = dm.get_data()  # lfc = (2 * lfc) + 1

    dm.clear_transformations()
    dm.clear_data()
    df0 = dm.get_data()  # lfc = lfc
    assert all((df0["lfc"].values + 1.0) == df1["lfc"].values)
    assert all(((2.0 * df0["lfc"].values) + 1.0) == df2["lfc"].values)


def test_get_transformations(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(
        depmap_test_data, transformations=[double_log_fold_change]
    )
    assert len(dm.get_transformations()) == 2
    dm.clear_transformations()
    assert len(dm.get_transformations()) == 0
    dm.set_transformations([double_log_fold_change, add_one_to_log_fold_change])
    assert len(dm.get_transformations()) == 2
    dm.insert_transformation(reverse_sgrna, at=1)
    assert len(dm.get_transformations()) == 3
    dm.clear_transformations()
    assert len(dm.get_transformations()) == 0


def test_mock_data_generation(depmap_test_data: Path) -> None:
    dm = CrisprScreenDataManager(data_source=depmap_test_data)
    small_mock_data = dm.generate_mock_data(size="small")
    medium_mock_data = dm.generate_mock_data(size="medium")
    large_mock_data = dm.generate_mock_data(size="large")
    assert (
        small_mock_data.shape[1]
        == medium_mock_data.shape[1]
        == large_mock_data.shape[1]
    )
    assert (
        small_mock_data.shape[0] < medium_mock_data.shape[0] < large_mock_data.shape[0]
    )
