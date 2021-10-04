#!/usr/bin/env python3
from itertools import product
from pathlib import Path
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from hypothesis.strategies import DataObject

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp

DATA_PATH = Path("tests", "depmap_test_data.csv")


@pytest.fixture
def data() -> pd.DataFrame:
    return achelp.read_achilles_data(DATA_PATH, low_memory=True)


#### ---- Reading and modifying Achilles data ---- ####


class TestHandlingAchillesData:
    def setup_class(self) -> None:
        self.file_path = DATA_PATH

    def test_reading_data(self) -> None:
        df = achelp.read_achilles_data(self.file_path, low_memory=True)
        required_cols = ["sgrna", "depmap_id", "hugo_symbol", "lfc", "copy_number"]
        for col in required_cols:
            assert col in df.columns

    def test_factor_columns_exist(self) -> None:
        test_data = achelp.read_achilles_data(
            self.file_path, low_memory=True, set_categorical_cols=False
        )
        assert "category" not in test_data.dtypes


class TestModifyingAchillesData:
    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        np.random.seed(0)
        genes = np.random.choice(list(LETTERS), 5, replace=False)
        n_measures = 100
        return pd.DataFrame(
            {
                "hugo_symbol": np.repeat(genes, n_measures),
                "copy_number": np.random.uniform(0, 100, len(genes) * n_measures),
            }
        )

    def test_reading_data(self, data: pd.DataFrame) -> None:
        assert "sgrna" in data.columns

    def test_factor_columns_exist(self, data: pd.DataFrame) -> None:
        required_factor_cols = ["sgrna", "depmap_id", "hugo_symbol"]
        for col in required_factor_cols:
            assert data[col].dtype == "category"

    def test_setting_achilles_categorical_columns(self) -> None:
        data = achelp.read_achilles_data(DATA_PATH, set_categorical_cols=False)
        assert "category" not in data.dtypes
        data = achelp.set_achilles_categorical_columns(data=data)
        assert sum(data.dtypes == "category") == 7

    def test_custom_achilles_categorical_columns(self) -> None:
        data = achelp.read_achilles_data(DATA_PATH, set_categorical_cols=False)
        data = achelp.set_achilles_categorical_columns(
            data, cols=["hugo_symbol", "sgrna"]
        )
        assert sum(data.dtypes == "category") == 2

    def test_subsample_data_reduces_size(self, data: pd.DataFrame) -> None:
        sub_data = achelp.subsample_achilles_data(data, n_genes=2)
        assert (
            data.shape[0] > sub_data.shape[0] >= 2
            and data.shape[1] == sub_data.shape[1]
        )

    def test_subsample_data_correct_number_of_genes(self, data: pd.DataFrame) -> None:
        for n in np.random.randint(2, 10, 100):
            sub_data = achelp.subsample_achilles_data(
                data, n_genes=n, n_cell_lines=None
            )
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_subsample_data_correct_number_of_cells(self, data: pd.DataFrame) -> None:
        for n in np.random.randint(2, 6, 100):
            sub_data = achelp.subsample_achilles_data(
                data, n_genes=None, n_cell_lines=n
            )
            assert n == len(sub_data["depmap_id"].unique())
            assert data.shape[0] > sub_data.shape[0] >= n

    def test_negative_subsamples(self, data: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            _ = achelp.subsample_achilles_data(data, n_genes=0)
            _ = achelp.subsample_achilles_data(data, n_genes=-1)
            _ = achelp.subsample_achilles_data(data, n_cell_lines=0)
            _ = achelp.subsample_achilles_data(data, n_cell_lines=-1)

    def test_z_scaling_means(self, mock_data: pd.DataFrame) -> None:
        print(mock_data)
        z_data = achelp.zscale_cna_by_group(mock_data)
        print(z_data)
        for gene in z_data["hugo_symbol"].unique():
            m = z_data.query(f"hugo_symbol == '{gene}'")["copy_number_z"].mean()
            assert m == pytest.approx(0.0, abs=0.01)

    def test_z_scaling_stddevs(self, mock_data: pd.DataFrame) -> None:
        z_data = achelp.zscale_cna_by_group(mock_data)
        for gene in z_data["hugo_symbol"].unique():
            s = z_data.query(f"hugo_symbol == '{gene}'")["copy_number_z"].std()
            assert s == pytest.approx(1.0, abs=0.1)

    def test_z_scaling_max(self, mock_data: pd.DataFrame) -> None:
        cn_max = 25
        z_data_max = achelp.zscale_cna_by_group(mock_data, cn_max=cn_max)
        for gene in z_data_max["hugo_symbol"].unique():
            m = z_data_max.query(f"hugo_symbol == '{gene}'")["copy_number_z"].mean()
            assert m == pytest.approx(0.0, abs=0.01)
            s = z_data_max.query(f"hugo_symbol == '{gene}'")["copy_number_z"].std()
            assert s == pytest.approx(1.0, abs=0.1)


@st.composite
def my_arrays(draw: Callable, min_size: int = 1) -> np.ndarray:
    return np.array(
        draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=100.0,
                    allow_infinity=False,
                    allow_nan=False,
                ),
                min_size=min_size,
            )
        )
    )


@given(my_arrays())
def test_zscale_rna_expression(rna_expr_ary: np.ndarray) -> None:
    df = pd.DataFrame({"rna_expr": rna_expr_ary})
    note(df)
    df_z = achelp.zscale_rna_expression(df, "rna_expr", new_col="rna_expr_z")
    assert df_z["rna_expr_z"].mean() == pytest.approx(0.0, abs=0.01)


@given(
    rna_expr_ary=my_arrays(),
    lower_bound=st.floats(
        min_value=-5.0, max_value=-0.01, allow_infinity=False, allow_nan=False
    ),
    upper_bound=st.floats(
        min_value=0.01, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_zscale_rna_expression_with_bounds(
    rna_expr_ary: list[float], lower_bound: float, upper_bound: float
) -> None:
    df = pd.DataFrame({"rna_expr": rna_expr_ary})
    note(df)
    df_z = achelp.zscale_rna_expression(
        df,
        "rna_expr",
        new_col="rna_expr_z",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    assert df_z["rna_expr_z"].mean() == pytest.approx(0.0, abs=1)
    assert df_z["rna_expr_z"].min() >= lower_bound
    assert df_z["rna_expr_z"].max() <= upper_bound


@given(st.data())
@settings(settings.get_profile("slow-adaptive"))
def test_zscale_rna_expression_by_gene_lineage(hyp_data: DataObject) -> None:
    n_lineages = hyp_data.draw(st.integers(min_value=1, max_value=4))
    n_genes = hyp_data.draw(st.integers(min_value=1, max_value=7))
    lineages = [f"lineage_{i}" for i in range(n_lineages)]
    genes = [f"gene_{i}" for i in range(n_genes)]
    df = pd.DataFrame(
        list(product(lineages, genes)), columns=["lineage", "hugo_symbol"]
    )
    df["rna_expr"] = np.random.uniform(0.0, 100.0, size=len(df))
    note(df.__str__())
    df_z = achelp.zscale_rna_expression(df, "rna_expr", new_col="rna_expr_z")
    assert df_z["rna_expr_z"].mean() == pytest.approx(0.0, abs=0.01)
    df_z_g = achelp.zscale_rna_expression_by_gene_lineage(
        df, "rna_expr", new_col="rna_expr_z"
    )
    for gene in genes:
        for lineage in lineages:
            rna_expr_z = df_z_g.query(f"hugo_symbol == '{gene}'").query(
                f"lineage == '{lineage}'"
            )["rna_expr_z"]
            assert np.mean(rna_expr_z) == pytest.approx(0, abs=0.01)


#### ---- Index helpers ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(letters), of_length, replace=True))


@pytest.fixture
def mock_gene_data() -> pd.DataFrame:
    genes = list(LETTERS[:5])
    gene_list: list[str] = []
    sgrna_list: list[str] = []
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
def example_achilles_data() -> pd.DataFrame:
    return achelp.read_achilles_data(Path("tests", "depmap_test_data.csv"))


def test_make_mapping_df() -> None:
    df = pd.DataFrame({"g1": np.arange(9), "g2": np.repeat(["a", "b", "c"], 3)})
    df_map = achelp.make_mapping_df(df, "g1", "g2")
    for i1, i2 in zip(df.itertuples(), df_map.itertuples()):
        assert i1.g1 == i2.g1
        assert i1.g2 == i2.g2

    df2 = pd.concat([df, df, df])
    df_map2 = achelp.make_mapping_df(df2, "g1", "g2")
    assert df.shape == df_map2.shape

    df3 = df.copy()
    df3["other_column"] = "X"
    df_map3 = achelp.make_mapping_df(df3, "g1", "g2")
    assert df.shape == df_map3.shape
    for i1, i2 in zip(df.itertuples(), df_map3.itertuples()):
        assert i1.g1 == i2.g1
        assert i1.g2 == i2.g2
    df_map3 = achelp.make_mapping_df(df3, "g1", "other_column")
    for i1, i2 in zip(df.itertuples(), df_map3.itertuples()):
        assert i1.g1 == i2.g1
        assert i2.other_column == "X"


def test_cell_line_to_lineage_map(example_achilles_data: pd.DataFrame) -> None:
    df_map = achelp.make_cell_line_to_lineage_mapping_df(example_achilles_data)
    assert "depmap_id" in df_map.columns
    assert "lineage" in df_map.columns
    assert len(df_map.depmap_id.unique()) == len(df_map)
    assert len(df_map.depmap_id.unique()) == len(
        example_achilles_data.depmap_id.unique()
    )
    assert len(df_map.lineage.unique()) == len(example_achilles_data.lineage.unique())


def test_sgrna_to_gene_mapping_df_is_smaller(mock_gene_data: pd.DataFrame) -> None:
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map) < len(mock_gene_data)
    assert sgrna_map["hugo_symbol"].dtype == "category"


def test_sgrna_to_gene_map_preserves_categories(mock_gene_data: pd.DataFrame) -> None:
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    for col in sgrna_map.columns:
        assert sgrna_map[col].dtype == "category"


def test_sgrna_are_unique(mock_gene_data: pd.DataFrame) -> None:
    sgrna_map = achelp.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map["sgrna"].values) == len(sgrna_map["sgrna"].values.unique())


def test_different_colnames(mock_gene_data: pd.DataFrame) -> None:
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


def test_common_idx_creation(example_achilles_data: pd.DataFrame) -> None:
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    isinstance(indices, achelp.CommonIndices)


def test_common_idx_counters(example_achilles_data: pd.DataFrame) -> None:
    indices = achelp.common_indices(example_achilles_data)
    assert indices.n_sgrnas == dphelp.nunique(indices.sgrna_idx)
    assert indices.n_genes == dphelp.nunique(indices.gene_idx)
    assert indices.n_celllines == dphelp.nunique(indices.cellline_idx)


def test_common_idx_sgrna_to_gene_map(example_achilles_data: pd.DataFrame) -> None:
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    for sgrna in example_achilles_data.sgrna.values:
        assert sgrna in indices.sgrna_to_gene_map.sgrna.values
    for gene in example_achilles_data.hugo_symbol.values:
        assert gene in indices.sgrna_to_gene_map.hugo_symbol.values


def test_common_idx_depmap(example_achilles_data: pd.DataFrame) -> None:
    indices = achelp.common_indices(example_achilles_data.sample(frac=1.0))
    assert dphelp.nunique(example_achilles_data.depmap_id.values) == dphelp.nunique(
        indices.cellline_idx
    )


def test_make_kras_mutation_index_with_other() -> None:
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


def test_make_kras_mutation_index_with_other_colnames() -> None:
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


# def test_uncommon_indices(example_achilles_data: pd.DataFrame) -> None:
#     idx = achelp.uncommon_indices(example_achilles_data)
#     assert idx.n_kras_mutations == len(
#         example_achilles_data["kras_mutation"].unique()
#     )
#     assert len(idx.cellline_to_kras_mutation_idx) == len(
#         example_achilles_data["depmap_id"].unique()
#     )

#     idx = achelp.uncommon_indices(example_achilles_data, min_kras_muts=5)
#     assert idx.n_kras_mutations < len(example_achilles_data["kras_mutation"].unique())
#     assert len(idx.cellline_to_kras_mutation_idx) == len(
#         example_achilles_data["depmap_id"].unique()
#     )


def test_data_batch_indices(example_achilles_data: pd.DataFrame) -> None:
    bi = achelp.data_batch_indices(example_achilles_data)
    n_sources = len(example_achilles_data["screen"].values.unique())
    n_batches = len(example_achilles_data["p_dna_batch"].values.unique())
    assert bi.n_screens == n_sources
    assert bi.n_batches == n_batches

    batch_map = bi.batch_to_screen_map
    np.testing.assert_array_equal(
        batch_map["p_dna_batch"].values, batch_map["p_dna_batch"].unique()
    )
