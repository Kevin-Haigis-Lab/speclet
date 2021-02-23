#!/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import common_data_processing as dphelp

#### ---- nunique ---- ####


class TestNunique:
    def test_nunique_string(self):
        assert dphelp.nunique("abc") == 1

    def test_nunique_list_int(self):
        assert dphelp.nunique([1, 3, 2, 1]) == 3

    def test_nunique_tuple(self):
        assert dphelp.nunique((1, 2, 3, 1, 3)) == 3

    def test_nunique_dict(self):
        d = {"a": 1, "b": 2, "c": 3}
        with pytest.raises(TypeError):
            dphelp.nunique(d)


#### ---- nmutations_to_binary_array ---- ####


# class TestNMutationsToBinaryArray:
#     def


#### ---- Reading and modifying Achilles data ---- ####

# TODO: In the original subsample script, make a small testing data frame.


class TestHandlingAchillesData:
    def setup(self):
        self.file_path = Path("tests", "depmap_test_data.csv")

    def test_reading_data(self):
        df = dphelp.read_achilles_data(self.file_path, low_memory=True)
        required_cols = ["sgrna", "depmap_id", "hugo_symbol", "lfc", "gene_cn"]
        for col in required_cols:
            assert col in df.columns

    def test_factor_columns_exist(self):
        test_data = dphelp.read_achilles_data(
            self.file_path, low_memory=True, set_categorical_cols=False
        )
        assert not "category" in test_data.dtypes


class TestModifyingAchillesData:
    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        return dphelp.read_achilles_data(
            Path("tests", "depmap_test_data.csv"), low_memory=True
        )

    def test_reading_data(self, test_data: pd.DataFrame):
        assert "sgrna" in test_data.columns

    def test_factor_columns_exist(self, test_data: pd.DataFrame):
        required_factor_cols = ["sgrna", "depmap_id", "hugo_symbol"]
        for col in required_factor_cols:
            assert test_data[col].dtype == "category"
