#!/usr/bin/env python3

from pathlib import Path
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp

DATA_PATH = Path("tests", "depmap_test_data.csv")


#### ---- Reading and modifying Achilles data ---- ####


class TestHandlingAchillesData:
    def setup_class(self):
        self.file_path = DATA_PATH

    def test_reading_data(self):
        df = achelp.read_achilles_data(self.file_path, low_memory=True)
        required_cols = ["sgrna", "depmap_id", "hugo_symbol", "lfc", "gene_cn"]
        for col in required_cols:
            assert col in df.columns

    def test_factor_columns_exist(self):
        test_data = achelp.read_achilles_data(
            self.file_path, low_memory=True, set_categorical_cols=False
        )
        assert "category" not in test_data.dtypes


class TestModifyingAchillesData:
    def setup_class(self):
        self.data_path = DATA_PATH

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        return achelp.read_achilles_data(self.data_path, low_memory=True)

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
        data = achelp.read_achilles_data(self.data_path, set_categorical_cols=False)
        assert "category" not in data.dtypes
        data = achelp.set_achilles_categorical_columns(data=data)
        assert sum(data.dtypes == "category") == 7

    def test_custom_achilles_categorical_columns(self):
        data = achelp.read_achilles_data(self.data_path, set_categorical_cols=False)
        data = achelp.set_achilles_categorical_columns(
            data, cols=["hugo_symbol", "sgrna"]
        )
        assert sum(data.dtypes == "category") == 2

    def test_subsample_data_reduces_size(self, data: pd.DataFrame):
        sub_data = achelp.subsample_achilles_data(data, n_genes=2)
        assert (
            data.shape[0] > sub_data.shape[0] >= 2
            and data.shape[1] == sub_data.shape[1]
        )

    def test_subsample_data_correct_number_of_genes(self, data: pd.DataFrame):
        for n in np.random.randint(2, 20, 100):
            sub_data = achelp.subsample_achilles_data(
                data, n_genes=n, n_cell_lines=None
            )
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_subsample_data_correct_number_of_cells(self, data: pd.DataFrame):
        for n in np.random.randint(2, 10, 100):
            sub_data = achelp.subsample_achilles_data(
                data, n_genes=None, n_cell_lines=n
            )
            assert n == len(sub_data["depmap_id"].unique())
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_negative_subsamples(self, data: pd.DataFrame):
        with pytest.raises(ValueError):
            _ = achelp.subsample_achilles_data(data, n_genes=0)
            _ = achelp.subsample_achilles_data(data, n_genes=-1)
            _ = achelp.subsample_achilles_data(data, n_cell_lines=0)
            _ = achelp.subsample_achilles_data(data, n_cell_lines=-1)

    def test_z_scaling_means(self, mock_data: pd.DataFrame):
        z_data = achelp.zscale_cna_by_group(mock_data)
        gene_means = z_data.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.mean).values
        expected_mean = np.zeros_like(gene_means)
        np.testing.assert_almost_equal(gene_means, expected_mean)

    def test_z_scaling_stddevs(self, mock_data: pd.DataFrame):
        z_data = achelp.zscale_cna_by_group(mock_data)
        gene_sds = z_data.groupby(["hugo_symbol"])["gene_cn_z"].agg(np.std).values
        expected_sd = np.ones_like(gene_sds)
        np.testing.assert_almost_equal(gene_sds, expected_sd, decimal=2)

    def test_z_scaling_max(self, mock_data: pd.DataFrame):
        cn_max = 25
        z_data = achelp.zscale_cna_by_group(mock_data)
        z_data_max = achelp.zscale_cna_by_group(mock_data, cn_max=cn_max)

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


#### ---- Index helpers ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(letters), of_length, replace=True))


@pytest.fixture
def mock_gene_data() -> pd.DataFrame:
    genes = list(LETTERS[:5])
    gene_list: List[str] = []
    sgrna_list: List[str] = []
    for gene in genes:
        for _ in range(np.random.randint(5, 10)):
            gene_list.append(gene)
            sgrna_list.append(make_mock_sgrna(20))

    df = pd.DataFrame({"hugo_symbol": gene_list, "sgrna": sgrna_list}, dtype="category")
    df = pd.concat([df] + [df.sample(frac=0.75) for _ in range(10)])
    df = df.sample(frac=1.0)
    df["y"] = np.random.randn(len(df))
    return df


@pytest.fixture(scope="module")
def example_achilles_data():
    return achelp.read_achilles_data(Path("tests", "depmap_test_data.csv"))


def test_sgrna_to_gene_mapping_df_is_smaller(mock_gene_data: pd.DataFrame):
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map) < len(mock_gene_data)
    assert sgrna_map["hugo_symbol"].dtype == "category"


def test_sgrna_to_gene_map_preserves_categories(mock_gene_data: pd.DataFrame):
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    for col in sgrna_map.columns:
        assert sgrna_map[col].dtype == "category"


def test_sgrna_are_unique(mock_gene_data: pd.DataFrame):
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map["sgrna"].values) == len(sgrna_map["sgrna"].values.unique())


def test_different_colnames(mock_gene_data: pd.DataFrame):
    df = mock_gene_data.rename(columns={"sgrna": "a", "hugo_symbol": "b"})
    sgrna_map_original = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    sgrna_map_new = achelp.make_sgrna_to_gene_mapping_df(
        df, sgrna_col="a", gene_col="b"
    )
    for col in ["a", "b"]:
        assert col in sgrna_map_new.columns
    for col_i in range(sgrna_map_new.shape[1]):
        np.testing.assert_array_equal(
            sgrna_map_new.iloc[:, col_i].values,
            sgrna_map_original.iloc[:, col_i].values,
        )


def test_common_idx_creation(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    isinstance(indices, achelp.CommonIndices)


def test_common_idx_counters(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data)
    assert indices.n_sgrnas == dphelp.nunique(indices.sgrna_idx)
    assert indices.n_genes == dphelp.nunique(indices.gene_idx)
    assert indices.n_celllines == dphelp.nunique(indices.cellline_idx)
    assert indices.n_batches == dphelp.nunique(indices.batch_idx)


def test_common_idx_sgrna_to_gene_map(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    for sgrna in example_achilles_data.sgrna.values:
        assert sgrna in indices.sgrna_to_gene_map.sgrna.values
    for gene in example_achilles_data.hugo_symbol.values:
        assert gene in indices.sgrna_to_gene_map.hugo_symbol.values


def test_common_idx_depmap(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    assert dphelp.nunique(example_achilles_data.depmap_id.values) == dphelp.nunique(
        indices.cellline_idx
    )


def test_common_idx_kras_mutation(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    assert dphelp.nunique(example_achilles_data.kras_mutation.values) == dphelp.nunique(
        indices.kras_mutation_idx
    )


def test_common_idx_pdna_batch(example_achilles_data: pd.DataFrame):
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    assert dphelp.nunique(example_achilles_data.pdna_batch.values) == dphelp.nunique(
        indices.batch_idx
    )


def test_make_kras_mutation_index_with_other():
    df = pd.DataFrame(
        {
            "depmap_id": ["a", "a", "b", "b", "c", "d", "d", "d", "e"],
            "kras_mutation": ["L", "L", "M", "M", "L", "N", "N", "N", "O"],
        }
    )
    real_idx = np.array([0, 0, 1, 1, 0, 2, 2, 2, 3])
    kras_idx = achelp.make_kras_mutation_index_with_other(df)
    np.testing.assert_array_equal(kras_idx, real_idx)

    real_idx = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])
    kras_idx = achelp.make_kras_mutation_index_with_other(df, min=2)
    np.testing.assert_array_equal(kras_idx, real_idx)


def test_make_kras_mutation_index_with_other_colnames():
    df = pd.DataFrame(
        {
            "cell_line": ["a", "a", "b", "b", "c", "d", "d", "d", "e"],
            "kras_allele": ["L", "L", "M", "M", "L", "N", "N", "N", "O"],
        }
    )

    with pytest.raises(ValueError):
        kras_idx = achelp.make_kras_mutation_index_with_other(df)

    real_idx = np.array([0, 0, 1, 1, 0, 2, 2, 2, 3])
    kras_idx = achelp.make_kras_mutation_index_with_other(
        df, kras_col="kras_allele", cl_col="cell_line"
    )
    np.testing.assert_array_equal(kras_idx, real_idx)

    real_idx = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])
    kras_idx = achelp.make_kras_mutation_index_with_other(
        df, min=2, kras_col="kras_allele", cl_col="cell_line"
    )
    np.testing.assert_array_equal(kras_idx, real_idx)


def test_uncommon_indices(example_achilles_data: pd.DataFrame):
    idx = achelp.uncommon_indices(example_achilles_data)
    assert idx.n_kras_mutations == len(example_achilles_data["kras_mutation"].unique())
    assert len(idx.cellline_to_kras_mutation_idx) == len(
        example_achilles_data["depmap_id"].unique()
    )

    idx = achelp.uncommon_indices(example_achilles_data, min_kras_muts=5)
    assert idx.n_kras_mutations < len(example_achilles_data["kras_mutation"].unique())
    assert len(idx.cellline_to_kras_mutation_idx) == len(
        example_achilles_data["depmap_id"].unique()
    )
