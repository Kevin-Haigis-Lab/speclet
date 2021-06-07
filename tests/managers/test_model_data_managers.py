#!/usr/bin/env python3

import pandas as pd
import seaborn as sns

from src.managers.model_data_managers import CrcDataManager

#### ---- Test CrcDataManager ---- ####


class TestCrcDataManager:
    def test_batch_size(self):
        dm = CrcDataManager()
        not_debug_batch_size = dm.get_batch_size()
        dm.debug = True
        debug_batch_size = dm.get_batch_size()
        assert debug_batch_size < not_debug_batch_size

    def test_data_paths(self):
        dm = CrcDataManager(debug=True)
        assert dm.get_data_path().exists and dm.get_data_path().is_file()

    def test_get_data(self):
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

    def test_data_transformations(self):
        dm = CrcDataManager(debug=True)
        dm.data = sns.load_dataset("iris")
        assert dm.data.shape[0] > 5

        def head(df: pd.DataFrame) -> pd.DataFrame:
            return df.head(n=5)

        def select(df: pd.DataFrame) -> pd.DataFrame:
            return df[["sepal_length", "species"]]

        dm.add_transformations([head])
        dm.data = sns.load_dataset("iris")
        assert dm.data.shape[0] == 5

        dm.add_transformations([select])
        dm.data = sns.load_dataset("iris")
        assert dm.data.shape[0] == 5
        assert dm.data.shape[1] == 2
        assert set(["sepal_length", "species"]) == set(dm.data.columns.to_list())
