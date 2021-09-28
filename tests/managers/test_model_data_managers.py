#!/usr/bin/env python3

from pathlib import Path
from random import choice
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
    monkeypatch.setattr(CrcDataManager, "_filter_for_broad_source_only", identity)


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
        assert {"sepal_length", "species"} == set(dm.data.columns.to_list())

    def test_transform_when_data_is_set(
        self,
        iris: pd.DataFrame,
        no_standard_crc_transformations,
        no_setting_achilles_categorical_columns,
    ):
        dm = CrcDataManager(debug=True)
        dm.add_transformations([head, select])
        assert dm.data is None
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

    @pytest.mark.parametrize("col", achilles._default_achilles_categorical_cols)
    def test_achilles_cat_columns_reset_when_data_is_retrieved(
        self, mock_crc_dm: CrcDataManager, col: str, monkeypatch: pytest.MonkeyPatch
    ):
        original_df = mock_crc_dm.get_data()
        mock_crc_dm.data = None
        mod_df = remove_random_cat(original_df, col=col)
        assert not check_achilles_cat_columns_correct_indexing(mod_df, col)

        def return_mod_data(*args: Any, **kwargs: Any) -> pd.DataFrame:
            return mod_df

        monkeypatch.setattr(mock_crc_dm, "_load_data", return_mod_data)
        new_data = mock_crc_dm.get_data()
        assert check_achilles_cat_columns_correct_indexing(new_data, col)

    @pytest.mark.parametrize("col", achilles._default_achilles_categorical_cols)
    def test_achilles_cat_columns_reset_when_data_is_assigned(
        self, mock_crc_dm: CrcDataManager, col: str
    ):
        original_df = mock_crc_dm.get_data()
        mod_df = remove_random_cat(original_df, col=col)
        assert not check_achilles_cat_columns_correct_indexing(mod_df, col)

        mock_crc_dm.data = mod_df
        assert mock_crc_dm.data is not None
        assert check_achilles_cat_columns_correct_indexing(mock_crc_dm.data, col)
        assert check_achilles_cat_columns_correct_indexing(mock_crc_dm.get_data(), col)

    @pytest.mark.parametrize("col", achilles._default_achilles_categorical_cols)
    def test_achilles_cat_columns_reset_when_data_is_set(
        self, mock_crc_dm: CrcDataManager, col: str
    ):
        original_df = mock_crc_dm.get_data()
        mod_df = remove_random_cat(original_df, col=col)
        assert not check_achilles_cat_columns_correct_indexing(mod_df, col)

        mock_crc_dm.set_data(mod_df)
        assert mock_crc_dm.data is not None
        assert check_achilles_cat_columns_correct_indexing(mock_crc_dm.data, col)
        assert check_achilles_cat_columns_correct_indexing(mock_crc_dm.get_data(), col)

    @pytest.mark.parametrize("col", achilles._default_achilles_categorical_cols)
    def test_achilles_cat_columns_reset_when_apply_transforms(
        self, mock_crc_dm: CrcDataManager, col: str
    ):
        original_df = mock_crc_dm.get_data()
        mod_df = remove_random_cat(original_df, col=col)
        assert not check_achilles_cat_columns_correct_indexing(mod_df, col)

        mock_crc_dm.transformations = []
        transformed_data = mock_crc_dm.apply_transformations(mod_df)
        assert check_achilles_cat_columns_correct_indexing(transformed_data, col)

    @pytest.mark.parametrize("col", achilles._default_achilles_categorical_cols)
    def test_achilles_cat_columns_reset_when_add_new_transforms(
        self, mock_crc_dm: CrcDataManager, col: str
    ):
        original_df = mock_crc_dm.get_data()

        def my_transformation(df: pd.DataFrame) -> pd.DataFrame:
            return remove_random_cat(df, col=col)

        mock_crc_dm.add_transformations([my_transformation], new_only=False)
        assert mock_crc_dm.data is not None
        assert original_df.shape[0] > mock_crc_dm.data.shape[0]
        assert check_achilles_cat_columns_correct_indexing(mock_crc_dm.data, col)

    @pytest.mark.DEV
    def test_filter_only_broad(
        self, monkeypatch: pytest.MonkeyPatch, depmap_test_data: Path
    ):
        def monkey_get_data_path(*args, **kwargs) -> Path:
            return depmap_test_data

        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(broad_only=False)
        data = dm.get_data()
        assert data["screen"].nunique() >= 1
        dm2 = CrcDataManager(broad_only=True)
        data = dm2.get_data()
        assert data["screen"].nunique() == 1
        assert data["screen"].unique()[0] == "broad"


def remove_random_cat(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove a random value from a column of a pandas data frame."""
    x = choice(df[col].tolist())
    mod_df = df.copy()[df[col] != x]
    mod_df = mod_df.reset_index(drop=True)
    return mod_df


def check_achilles_cat_columns_correct_indexing(data: pd.DataFrame, col: str) -> bool:
    check_one = data[col].nunique() == len(data[col].cat.categories)
    check_two = set(range(data[col].nunique())) == set(data[col].cat.codes.tolist())
    return check_one and check_two
