#!/bin/env python3

from pathlib import Path
from string import ascii_uppercase as LETTERS

import numpy as np
import pandas as pd
import pytest
from numpy.random import uniform
from pandas.core.algorithms import mode

from analysis import common_data_processing as dphelp

DATA_PATH = Path("tests", "depmap_test_data.csv")

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


#### ---- Reading and modifying Achilles data ---- ####


class TestHandlingAchillesData:
    def setup_class(self):
        self.file_path = DATA_PATH

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
    def setup_class(self):
        self.data_path = DATA_PATH

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        return dphelp.read_achilles_data(self.data_path, low_memory=True)

    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        np.random.seed(0)
        genes = np.random.choice(list(LETTERS), 5, replace=False)
        n_measures = 100
        return pd.DataFrame(
            {
                "hugo_symbol": np.repeat(genes, n_measures),
                "gene_cn": np.random.uniform(0, 100, len(genes) * n_measures),
            }
        )

    def test_reading_data(self, data: pd.DataFrame):
        assert "sgrna" in data.columns

    def test_factor_columns_exist(self, data: pd.DataFrame):
        required_factor_cols = ["sgrna", "depmap_id", "hugo_symbol"]
        for col in required_factor_cols:
            assert data[col].dtype == "category"

    def test_setting_achilles_categorical_columns(self):
        data = dphelp.read_achilles_data(self.data_path, set_categorical_cols=False)
        assert not "category" in data.dtypes
        data = dphelp.set_achilles_categorical_columns(data=data)
        assert sum(data.dtypes == "category") == 5

    def test_custom_achilles_categorical_columns(self):
        data = dphelp.read_achilles_data(self.data_path, set_categorical_cols=False)
        data = dphelp.set_achilles_categorical_columns(
            data, cols=["hugo_symbol", "sgrna"]
        )
        assert sum(data.dtypes == "category") == 2

    def test_subsample_data_reduces_size(self, data: pd.DataFrame):
        sub_data = dphelp.subsample_achilles_data(data, n_genes=2)
        assert (
            data.shape[0] > sub_data.shape[0] >= 2
            and data.shape[1] == sub_data.shape[1]
        )

    def test_subsample_data_correct_number_of_genes(self, data: pd.DataFrame):
        for n in np.random.randint(2, 20, 100):
            sub_data = dphelp.subsample_achilles_data(data, n_genes=n)
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_subsample_data_correct_number_of_cells(self, data: pd.DataFrame):
        for n in np.random.randint(2, 20, 100):
            sub_data = dphelp.subsample_achilles_data(data, n_cell_lines=n)
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_negative_subsamples(self, data: pd.DataFrame):
        with pytest.raises(ValueError):
            _ = dphelp.subsample_achilles_data(data, n_genes=0)
            _ = dphelp.subsample_achilles_data(data, n_genes=-1)
            _ = dphelp.subsample_achilles_data(data, n_cell_lines=0)
            _ = dphelp.subsample_achilles_data(data, n_cell_lines=-1)

    def test_z_scaling_means(self, mock_data: pd.DataFrame):
        z_data = dphelp.zscale_cna_by_group(mock_data)
        gene_means = z_data.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.mean).values
        expected_mean = np.zeros_like(gene_means)
        np.testing.assert_almost_equal(gene_means, expected_mean)

    def test_z_scaling_stddevs(self, mock_data: pd.DataFrame):
        z_data = dphelp.zscale_cna_by_group(mock_data)
        gene_sds = z_data.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.std).values
        expected_sd = np.ones_like(gene_sds)
        np.testing.assert_almost_equal(gene_sds, expected_sd, decimal=2)

    def test_z_scaling_max(self, mock_data: pd.DataFrame):
        cn_max = 25
        z_data = dphelp.zscale_cna_by_group(mock_data)
        z_data_max = dphelp.zscale_cna_by_group(mock_data, cn_max=cn_max)

        gene_means = (
            z_data_max.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.mean).values
        )
        expected_mean = np.zeros_like(gene_means)
        np.testing.assert_almost_equal(gene_means, expected_mean)

        gene_sds = z_data_max.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.std).values
        expected_sd = np.ones_like(gene_sds)
        np.testing.assert_almost_equal(gene_sds, expected_sd, decimal=2)

        less_than_max_idx = np.where(mock_data["gene_cn"] < cn_max)
        z_vals = z_data.loc[less_than_max_idx, "gene_cn_z"]
        z_max_vals = z_data_max.loc[less_than_max_idx, "gene_cn_z"]
        assert (z_vals <= z_max_vals).all()


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
