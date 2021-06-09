from pathlib import Path
from string import ascii_letters
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from hypothesis import given
from hypothesis import strategies as st

from src.data_processing import common as dphelp
from src.modeling import simulation_based_calibration_helpers as sbc

chars = list(ascii_letters) + [str(i) for i in (range(10))]


class TestSBCFileManager:
    def test_init(self, tmp_path: Path):
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert not fm.all_data_exists()

    @pytest.fixture()
    def priors(self) -> Dict[str, Any]:
        return {
            "alpha": np.random.uniform(0, 100, size=3),
            "beta_log": np.random.uniform(0, 100, size=(10, 15)),
        }

    @pytest.fixture
    def posterior_summary(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [5, 6, 7], "y": ["a", "b", "c"]})

    def test_saving(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)
        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()

    def test_reading(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)

        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()
        read_results = fm.get_sbc_results()
        assert isinstance(read_results, sbc.SBCResults)
        assert isinstance(read_results.inference_obj, az.InferenceData)
        for k in read_results.priors:
            np.testing.assert_array_equal(read_results.priors[k], priors[k])

        for c in read_results.posterior_summary.columns:
            np.testing.assert_array_equal(
                read_results.posterior_summary[c].values, posterior_summary[c].values
            )


#### ---- Test mock data generation ---- ####


@given(st.integers(1, 50), st.integers(1, 5))
def test_generate_mock_sgrna_gene_map(n_genes: int, n_sgrnas_per_gene: int):
    sgrna_gene_map = sbc.generate_mock_sgrna_gene_map(
        n_genes=n_genes, n_sgrnas_per_gene=n_sgrnas_per_gene
    )
    assert len(sgrna_gene_map["hugo_symbol"].unique()) == n_genes
    assert len(sgrna_gene_map["sgrna"].unique()) == int(n_sgrnas_per_gene * n_genes)
    sgrnas = sgrna_gene_map["sgrna"].values
    assert len(sgrnas) == len(np.unique(sgrnas))


@given(st.data())
def test_generate_mock_cell_line_information(data: st.DataObject):
    genes = data.draw(
        st.lists(
            st.text().map(lambda s: s.encode("ascii", "ignore")),
            min_size=1,
            unique=True,
        ),
        label="genes",
    )
    genes = [str(g) for g in genes]
    n_cell_lines = data.draw(st.integers(1, 10), label="n_cell_lines")
    n_lineages = data.draw(st.integers(1, n_cell_lines), label="n_lineages")
    n_batches = data.draw(st.integers(1, n_cell_lines), label="n_batches")
    n_screens = data.draw(st.integers(1, n_cell_lines), label="n_screens")
    mock_info = sbc.generate_mock_cell_line_information(
        genes=genes,
        n_cell_lines=n_cell_lines,
        n_batches=n_batches,
        n_lineages=n_lineages,
        n_screens=n_screens,
    )
    assert len(mock_info["hugo_symbol"].unique()) == len(genes)
    assert len(mock_info["depmap_id"].unique()) == n_cell_lines
    assert len(mock_info["lineage"].unique()) <= n_lineages
    assert len(mock_info["screen"].unique()) <= n_screens
    assert len(mock_info["p_dna_batch"].unique()) <= n_batches


@st.composite
def generate_data_with_random_params(draw) -> pd.DataFrame:
    n_genes = draw(st.integers(2, 20), label="n_genes")
    n_sgrnas_per_gene = draw(st.integers(2, 20), label="n_sgrnas_per_gene")
    n_cell_lines = draw(st.integers(3, 20), label="n_cell_lines")
    n_lineages = draw(st.integers(1, n_cell_lines), label="n_lineages")
    n_batches = draw(st.integers(1, n_cell_lines), label="n_batches")
    if n_batches == 1:
        n_screens = 1
    else:
        n_screens = draw(st.integers(1, n_cell_lines), label="n_screens")

    return sbc.generate_mock_achilles_data(
        n_genes=n_genes,
        n_sgrnas_per_gene=n_sgrnas_per_gene,
        n_cell_lines=n_cell_lines,
        n_lineages=n_lineages,
        n_batches=n_batches,
        n_screens=n_screens,
    )


def test_add_mock_copynumber_data():
    df = sns.load_dataset("iris")
    df_cna = sbc.add_mock_copynumber_data(df)
    assert "copy_number" in df_cna.columns.to_list()
    assert all(df_cna["copy_number"] > 0.0)
    assert not any(df_cna["copy_number"].isna())


@given(
    st.floats(-5.0, 5.0, allow_infinity=False, allow_nan=False),
    st.floats(0.0, 2.0, allow_infinity=False, allow_nan=False),
)
def test_add_mock_zero_effect_lfc_data(mu: float, sigma: float):
    df = sns.load_dataset("iris")
    df_lfc = sbc.add_mock_zero_effect_lfc_data(df, mu=mu, sigma=sigma)
    assert "lfc" in df_lfc.columns.to_list()
    assert df_lfc["lfc"].mean() == pytest.approx(mu, abs=0.5)
    assert df_lfc["lfc"].std() == pytest.approx(sigma, abs=0.5)
    assert not any(df_lfc["lfc"].isna())


@given(st.data())
def test_mock_data_has_correct_categories_sizes(data):
    n_genes = data.draw(st.integers(2, 20), label="n_genes")
    n_sgrnas_per_gene = data.draw(st.integers(2, 20), label="n_sgrnas_per_gene")
    n_cell_lines = data.draw(st.integers(3, 20), label="n_cell_lines")
    n_lineages = data.draw(st.integers(1, n_cell_lines), label="n_lineages")
    n_batches = data.draw(st.integers(1, n_cell_lines), label="n_batches")
    if n_batches == 1:
        n_screens = 1
    else:
        n_screens = data.draw(st.integers(1, n_cell_lines), label="n_screens")

    mock_data = sbc.generate_mock_achilles_data(
        n_genes=n_genes,
        n_sgrnas_per_gene=n_sgrnas_per_gene,
        n_cell_lines=n_cell_lines,
        n_lineages=n_lineages,
        n_batches=n_batches,
        n_screens=n_screens,
    )
    assert n_genes == dphelp.nunique(mock_data.hugo_symbol)
    assert n_genes * n_sgrnas_per_gene == dphelp.nunique(mock_data.sgrna)
    assert n_cell_lines == dphelp.nunique(mock_data.depmap_id)
    assert n_batches >= dphelp.nunique(mock_data.p_dna_batch)


@given(mock_data=generate_data_with_random_params())
def test_sgrnas_uniquely_map_to_genes(mock_data: pd.DataFrame):
    sgrna_gene_map = (
        mock_data[["sgrna", "hugo_symbol"]].drop_duplicates().reset_index(drop=True)
    )
    sgrnas = sgrna_gene_map["sgrna"].values
    assert len(sgrnas) == len(np.unique(sgrnas))


@given(mock_data=generate_data_with_random_params())
def test_cellline_in_one_batch(mock_data: pd.DataFrame):
    cellline_to_batch = (
        mock_data[["depmap_id", "p_dna_batch"]].drop_duplicates().reset_index(drop=True)
    )
    cell_lines = cellline_to_batch["depmap_id"].values
    assert len(cell_lines) == len(np.unique(cell_lines))


@given(mock_data=generate_data_with_random_params())
def test_sgrna_for_each_cellline(mock_data: pd.DataFrame):
    all_sgrnas = set(mock_data.sgrna.values.to_list())
    for cell_line in mock_data.depmap_id.values.unique():
        cell_line_sgrnas = mock_data[
            mock_data.depmap_id == cell_line
        ].sgrna.values.to_list()
        # Confirm that each combo happens  exactly once.
        assert len(all_sgrnas) == len(cell_line_sgrnas)
        assert len(all_sgrnas.difference(set(cell_line_sgrnas))) == 0
