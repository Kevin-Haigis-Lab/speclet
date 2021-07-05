#!/usr/bin/env python3

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import seaborn as sns

from src.data_processing import achilles
from src.managers.model_data_managers import CrcDataManager

#### ---- Test CrcDataManager ---- ####


def head(df: pd.DataFrame) -> pd.DataFrame:
    return df.head(n=5)


def select(df: pd.DataFrame) -> pd.DataFrame:
    return df[["sepal_length", "species"]]


def mock_load_data(*args, **kwargs) -> pd.DataFrame:
    return sns.load_dataset("iris")


@pytest.fixture
def no_standard_crc_transformations(monkeypatch: pytest.MonkeyPatch):
    def identity(df: pd.DataFrame) -> pd.DataFrame:
        return df

    monkeypatch.setattr(
        CrcDataManager, "_drop_sgrnas_that_map_to_multiple_genes", identity
    )
    monkeypatch.setattr(CrcDataManager, "_drop_missing_copynumber", identity)


@pytest.fixture
def no_setting_achilles_categorical_columns(monkeypatch: pytest.MonkeyPatch):
    def identity(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return df

    monkeypatch.setattr(achilles, "set_achilles_categorical_columns", identity)


class TestCrcDataManager:
    @pytest.fixture(scope="function")
    def iris(self) -> pd.DataFrame:
        return sns.load_dataset("iris")

    def test_batch_size(self):
        dm = CrcDataManager()
        not_debug_batch_size = dm.get_batch_size()
        dm.debug = True
        debug_batch_size = dm.get_batch_size()
        assert debug_batch_size < not_debug_batch_size

    def test_data_paths(self):
        dm = CrcDataManager(debug=True)
        assert dm.get_data_path().exists and dm.get_data_path().is_file()
        assert dm.get_data_path().suffix == ".csv"

    def test_data_paths_not_debug(self):
        dm = CrcDataManager(debug=False)
        assert isinstance(dm.get_data_path(), Path)
        assert dm.get_data_path().suffix == ".csv"

    def test_set_data_to_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager(debug=True)
        assert dm.data is None

        # Monkeypatch to load 'iris' instead of real data.
        monkeypatch.setattr(CrcDataManager, "_load_data", mock_load_data)
        dm.get_data()
        assert dm.data is not None
        dm.data = None
        assert dm.data is None

        dm.get_data()
        assert dm.data is not None
        dm.set_data(None)
        assert dm.data is None

    def test_set_data_apply_trans(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager()
        dm.add_transformations([head])
        dm.set_data(iris)
        assert dm.data is not None and dm.data.shape[0] == 5
        dm.data = None
        dm.data = iris
        assert dm.data is not None and dm.data.shape[0] == 5

    def test_get_data(
        self, no_standard_crc_transformations, no_setting_achilles_categorical_columns
    ):
        dm = CrcDataManager(debug=True)
        assert dm.data is None
        data = dm.get_data()
        assert dm.data is not None
        assert dm.data.shape[0] > dm.data.shape[1]
        assert dm.data.shape[0] == data.shape[0]
        assert dm.data.shape[1] == data.shape[1]

    def test_mock_data_generation(self):
        dm = CrcDataManager(debug=True)
        small_mock_data = dm.generate_mock_data(size="small")
        medium_mock_data = dm.generate_mock_data(size="medium")
        large_mock_data = dm.generate_mock_data(size="large")
        assert (
            small_mock_data.shape[1]
            == medium_mock_data.shape[1]
            == large_mock_data.shape[1]
        )
        assert (
            small_mock_data.shape[0]
            < medium_mock_data.shape[0]
            < large_mock_data.shape[0]
        )

    def test_init_with_transformations(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        assert iris.shape[0] > 5
        dm = CrcDataManager(debug=True, transformations=[head])
        dm.data = iris
        assert dm.data.shape[0] == 5

    def test_data_transformations(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager(debug=True)
        dm.data = iris.copy()
        assert dm.data.shape[0] > 5

        dm.add_transformations([head])
        dm.data = iris.copy()
        assert dm.data.shape[0] == 5

        dm.add_transformations([select])
        dm.data = iris.copy()
        assert dm.data.shape[0] == 5
        assert dm.data.shape[1] == 2
        assert set(["sepal_length", "species"]) == set(dm.data.columns.to_list())

    def test_transform_when_data_is_set(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager(debug=True)
        dm.add_transformations([head, select])
        dm.data is None
        dm.data = iris
        assert dm.get_data().shape[0] == 5
        assert dm.get_data().shape[1] == 2

    def test_transform_when_getting_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        # Monkeypatch to load 'iris' instead of real data.
        monkeypatch.setattr(CrcDataManager, "_load_data", mock_load_data)

        dm = CrcDataManager(debug=True)
        dm.add_transformations([head, select])
        data = dm.get_data()
        assert data is dm.get_data()
        assert data.shape[0] == 5
        assert data.shape[1] == 2

    def test_adding_new_transformations(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager(debug=True)
        dm.data = iris.copy()
        dm.transform_data()
        assert dm.data.shape == iris.shape

        dm.add_transformations([head], run_transformations=False)
        assert dm.data.shape == iris.shape

        dm.add_transformations([select], run_transformations=True, new_only=True)
        assert dm.data.shape[0] == iris.shape[0]
        assert dm.data.shape[1] == 2

        dm.transform_data()
        assert dm.data.shape[0] == 5
        assert dm.data.shape[1] == 2
