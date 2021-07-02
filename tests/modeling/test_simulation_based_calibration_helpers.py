from pathlib import Path
from string import ascii_letters
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from hypothesis import given, settings
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

selection_methods = [a.value for a in sbc.SelectionMethod]


@pytest.mark.parametrize("method", selection_methods)
@given(
    n=st.integers(0, 100),
    list=st.lists(st.text(chars, min_size=1), min_size=1),
)
def test_select_n_elements_from_l_strings(method: str, n: int, list: List[str]):
    selection = sbc.select_n_elements_from_l(n, list, method=method)
    assert len(selection) == n
    for a in selection:
        assert a in list


@pytest.mark.parametrize("method", selection_methods)
@given(
    n=st.integers(0, 100),
    list=st.lists(st.integers(), min_size=1),
)
def test_select_n_elements_from_l_integers(method: str, n: int, list: List[int]):
    selection = sbc.select_n_elements_from_l(n, list, method=method)
    assert len(selection) == n
    for a in selection:
        assert a in list


def test_select_n_elements_from_l_tiled():
    list = ["j", "h", "c"]
    n = 5
    selection = sbc.select_n_elements_from_l(n, list, "tiled")
    assert len(selection) == n
    np.testing.assert_equal(selection, np.array(["j", "h", "c", "j", "h"]))


def test_select_n_elements_from_l_repeated():
    list = ["j", "h", "c"]
    n = 5
    selection = sbc.select_n_elements_from_l(n, list, "repeated")
    assert len(selection) == n
    np.testing.assert_equal(selection, np.array(["j", "j", "h", "h", "c"]))


@given(
    n=st.integers(5, 100),
    list=st.lists(st.text(chars, min_size=1), min_size=2, unique=True),
)
def test_select_n_elements_from_l_random_are_different(n: int, list: List[str]):
    n_same = 0
    for _ in range(10):
        a = sbc.select_n_elements_from_l(n, list, method="random")
        b = sbc.select_n_elements_from_l(n, list, method="random")
        n_same += all(a == b)
    assert n_same / 10 < 0.5


@given(
    n=st.integers(5, 100),
    list=st.lists(st.text(chars, min_size=1), min_size=2, unique=True),
)
def test_select_n_elements_from_l_shuffled_are_different(n: int, list: List[str]):
    n_same = 0
    for _ in range(10):
        a = sbc.select_n_elements_from_l(n, list, method="shuffled")
        b = sbc.select_n_elements_from_l(n, list, method="shuffled")
        n_same += all(a == b)
    assert n_same / 10 < 0.5


@given(
    n=st.integers(5, 100),
    list=st.lists(st.text(chars, min_size=1), min_size=2, unique=True),
)
def test_select_n_elements_from_l_shuffled_have_even_coverage(n: int, list: List[str]):
    a = sbc.select_n_elements_from_l(n, list, method="shuffled")
    b = sbc.select_n_elements_from_l(n, list, method="shuffled")
    for element in list:
        assert np.sum(a == element) == np.sum(b == element)


@given(st.integers(1, 50), st.integers(1, 5))
def test_generate_mock_sgrna_gene_map(n_genes: int, n_sgrnas_per_gene: int):
    sgrna_gene_map = sbc.generate_mock_sgrna_gene_map(
        n_genes=n_genes, n_sgrnas_per_gene=n_sgrnas_per_gene
    )
    assert len(sgrna_gene_map["hugo_symbol"].unique()) == n_genes
    assert len(sgrna_gene_map["sgrna"].unique()) == int(n_sgrnas_per_gene * n_genes)
    sgrnas = sgrna_gene_map["sgrna"].values
    assert len(sgrnas) == len(np.unique(sgrnas))


def test_generate_mock_cell_line_information():
    genes = [f"gene_{i}" for i in range(5)]
    n_cell_lines = 10
    n_lineages = 2
    n_batches = 2
    n_screens = 2
    mock_info = sbc.generate_mock_cell_line_information(
        genes=genes,
        n_cell_lines=n_cell_lines,
        n_batches=n_batches,
        n_lineages=n_lineages,
        n_screens=n_screens,
        randomness=False,
    )
    assert len(mock_info["hugo_symbol"].unique()) == len(genes)
    assert len(mock_info["depmap_id"].unique()) == n_cell_lines
    assert len(mock_info["lineage"].unique()) == n_lineages
    assert len(mock_info["screen"].unique()) == n_screens
    assert len(mock_info["p_dna_batch"].unique()) == n_batches


@given(st.data())
def test_generate_mock_cell_line_information_notrandomness(data: st.DataObject):
    genes = data.draw(st.lists(st.text(chars), min_size=1, unique=True), label="genes")
    n_cell_lines = data.draw(st.integers(5, 10), label="n_cell_lines")
    n_lineages = data.draw(st.integers(2, n_cell_lines), label="n_lineages")
    n_batches = data.draw(st.integers(2, n_cell_lines), label="n_batches")
    n_screens = (
        2 if n_batches == 2 else data.draw(st.integers(2, n_batches), label="n_screens")
    )
    mock_info = sbc.generate_mock_cell_line_information(
        genes=genes,
        n_cell_lines=n_cell_lines,
        n_batches=n_batches,
        n_lineages=n_lineages,
        n_screens=n_screens,
        randomness=False,
    )
    assert len(mock_info["hugo_symbol"].unique()) == len(genes)
    assert len(mock_info["depmap_id"].unique()) == n_cell_lines
    assert len(mock_info["lineage"].unique()) == n_lineages
    assert len(mock_info["screen"].unique()) == n_screens
    assert len(mock_info["p_dna_batch"].unique()) == n_batches


@given(st.data())
def test_generate_mock_cell_line_information_randomness(data: st.DataObject):
    genes = data.draw(
        st.lists(st.text(chars), min_size=1, max_size=25, unique=True), label="genes"
    )
    n_cell_lines = data.draw(st.integers(1, 10), label="n_cell_lines")
    n_lineages = data.draw(st.integers(1, min(n_cell_lines, 3)), label="n_lineages")
    n_batches = data.draw(st.integers(1, min(n_cell_lines, 3)), label="n_batches")
    n_screens = data.draw(st.integers(1, min(n_cell_lines, 3)), label="n_screens")
    mock_info = sbc.generate_mock_cell_line_information(
        genes=genes,
        n_cell_lines=n_cell_lines,
        n_batches=n_batches,
        n_lineages=n_lineages,
        n_screens=n_screens,
        randomness=False,
    )
    assert len(mock_info["hugo_symbol"].unique()) == len(genes)
    assert len(mock_info["depmap_id"].unique()) == n_cell_lines
    assert len(mock_info["lineage"].unique()) <= n_lineages
    assert len(mock_info["screen"].unique()) <= n_screens
    assert len(mock_info["p_dna_batch"].unique()) <= n_batches


@settings(deadline=60000)  # 60 seconds
@given(
    n_genes=st.integers(1, 5),
    n_sgrnas_per_gene=st.integers(1, 3),
    n_cell_lines=st.integers(2, 5),
    n_lineages=st.integers(1, 3),
    n_batches=st.integers(1, 3),
    n_screens=st.integers(1, 3),
    randomness=st.booleans(),
)
def test_generate_mock_achilles_categorical_groups(
    n_genes: int,
    n_sgrnas_per_gene: int,
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool,
):
    df = sbc.generate_mock_achilles_categorical_groups(
        n_genes=n_genes,
        n_sgrnas_per_gene=n_sgrnas_per_gene,
        n_cell_lines=n_cell_lines,
        n_lineages=n_lineages,
        n_batches=n_batches,
        n_screens=n_screens,
        randomness=randomness,
    )
    assert len(df["hugo_symbol"].cat.categories) == n_genes
    assert len(df["depmap_id"].cat.categories) == n_cell_lines
    for gene in df["hugo_symbol"].cat.categories:
        n_sgrna = (
            df.query(f"hugo_symbol == '{gene}'")[["hugo_symbol", "sgrna"]]
            .drop_duplicates()
            .shape[0]
        )
        assert n_sgrna == n_sgrnas_per_gene
        for cell_line in df["depmap_id"].cat.categories:
            n_rows = (
                df.query(f"hugo_symbol == '{gene}'")
                .query(f"depmap_id == '{cell_line}'")
                .shape[0]
            )
            assert n_rows > 0


@st.composite
def generate_data_with_random_params(draw) -> pd.DataFrame:
    n_genes = draw(st.integers(2, 7), label="n_genes")
    n_sgrnas_per_gene = draw(st.integers(1, 5), label="n_sgrnas_per_gene")
    n_cell_lines = draw(st.integers(3, 6), label="n_cell_lines")
    n_lineages = draw(st.integers(1, min(n_cell_lines, 3)), label="n_lineages")
    n_batches = draw(st.integers(1, min(n_cell_lines, 3)), label="n_batches")
    if n_batches == 1:
        n_screens = 1
    else:
        n_screens = draw(st.integers(1, min(n_cell_lines, 3)), label="n_screens")

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
    for _ in range(10):
        df_cna = sbc.add_mock_copynumber_data(df.copy())
        assert "copy_number" in df_cna.columns.to_list()
        assert all(df_cna["copy_number"] >= 0.0)
        assert not any(df_cna["copy_number"].isna())


def test_add_mock_rna_expression_data():
    df = sbc.generate_mock_achilles_categorical_groups(
        n_genes=5,
        n_sgrnas_per_gene=3,
        n_cell_lines=4,
        n_lineages=2,
        n_batches=1,
        n_screens=1,
    )
    mod_df = sbc.add_mock_rna_expression_data(df)
    assert "rna_expr" in mod_df.columns.to_list()
    assert np.all(mod_df["rna_expr"] >= 0.0)
    assert not np.any(mod_df["rna_expr"].isna())
    assert np.all(np.isfinite(mod_df["rna_expr"].values))


def test_add_mock_rna_expression_data_grouped():
    df = sbc.generate_mock_achilles_categorical_groups(
        n_genes=10,
        n_sgrnas_per_gene=100,
        n_cell_lines=4,
        n_lineages=2,
        n_batches=1,
        n_screens=1,
    )
    mod_df = sbc.add_mock_rna_expression_data(df, groups=["hugo_symbol"])
    assert "rna_expr" in mod_df.columns.to_list()
    assert np.all(mod_df["rna_expr"] >= 0.0)
    assert not np.any(mod_df["rna_expr"].isna())
    assert np.all(np.isfinite(mod_df["rna_expr"].values))

    gene_avgs = []
    for gene in mod_df["hugo_symbol"].cat.categories:
        gene_avg = (
            mod_df.query(f"hugo_symbol == '{gene}'")
            .groupby("depmap_id")["rna_expr"]
            .mean()
            .values
        )
        print(gene_avg)
        mean_error = np.mean(gene_avg - gene_avg.mean())
        assert mean_error == pytest.approx(0.0, abs=1.5)
        gene_avgs.append(gene_avg.mean())

    # The different groups have different means.
    assert not np.allclose(np.array(gene_avgs) - np.mean(gene_avgs), 0.0, atol=1.0)


@given(st.floats(0, 1, allow_nan=False, allow_infinity=False))
def test_add_mock_is_mutated_data(prob: float):
    df = pd.DataFrame({"A": np.zeros(10000)})
    df = sbc.add_mock_is_mutated_data(df, prob=prob)
    assert df["is_mutated"].mean() == pytest.approx(prob, abs=0.1)


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


@settings(deadline=None)
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


@settings(deadline=60000)  # 60 seconds
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
