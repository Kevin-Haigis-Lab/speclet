from pathlib import Path
from string import ascii_letters, ascii_lowercase
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
import seaborn as sns
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.data_processing import common as dphelp
from src.exceptions import IncorrectNumberOfFilesFoundError
from src.modeling import simulation_based_calibration_helpers as sbc

chars = list(ascii_letters) + [str(i) for i in (range(10))]


class TestSBCFileManager:
    def test_init(self, tmp_path: Path):
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert not fm.all_data_exists()

    @pytest.fixture()
    def priors(self) -> dict[str, Any]:
        return {
            "alpha": np.random.uniform(0, 100, size=3),
            "beta_log": np.random.uniform(0, 100, size=(10, 15)),
        }

    @pytest.fixture
    def posterior_summary(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [5, 6, 7], "y": ["a", "b", "c"]})

    @pytest.fixture
    def iris(self) -> pd.DataFrame:
        return sns.load_dataset("iris")

    def test_saving(
        self, tmp_path: Path, priors: dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)
        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()

    def test_reading(
        self, tmp_path: Path, priors: dict[str, Any], posterior_summary: pd.DataFrame
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

    def test_saving_simulation_dataframe(self, tmp_path: Path, iris: pd.DataFrame):
        fm = sbc.SBCFileManager(tmp_path)
        fm.save_sbc_data(iris)
        assert fm.get_sbc_data() is iris
        fm.sbc_data = None
        assert fm.get_sbc_data() is not iris
        assert fm.get_sbc_data().shape == iris.shape

    def test_clearing_saved_simulation_dataframe(
        self, tmp_path: Path, iris: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(tmp_path)
        fm.save_sbc_data(iris)
        assert fm.sbc_data_path.exists()
        fm.clear_results()
        assert fm.sbc_data_path.exists()
        fm.clear_saved_data()
        assert not fm.sbc_data_path.exists()


# class TestSBCAnalysis:


#### ---- Test mock data generation ---- ####

selection_methods = [a.value for a in sbc.SelectionMethod]


@pytest.mark.parametrize("method", selection_methods)
@given(
    n=st.integers(0, 100),
    list=st.lists(st.text(chars, min_size=1), min_size=1),
)
def test_select_n_elements_from_l_strings(method: str, n: int, list: list[str]):
    selection = sbc.select_n_elements_from_l(n, list, method=method)
    assert len(selection) == n
    for a in selection:
        assert a in list


@pytest.mark.parametrize("method", selection_methods)
@given(
    n=st.integers(0, 100),
    list=st.lists(st.integers(), min_size=1),
)
def test_select_n_elements_from_l_integers(method: str, n: int, list: list[int]):
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
def test_select_n_elements_from_l_random_are_different(n: int, list: list[str]):
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
def test_select_n_elements_from_l_shuffled_are_different(n: int, list: list[str]):
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
def test_select_n_elements_from_l_shuffled_have_even_coverage(n: int, list: list[str]):
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


@settings(settings.get_profile("slow-adaptive"))
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


def test_add_mock_copynumber_data_with_groups():
    df = sns.load_dataset("iris")
    for _ in range(10):
        df_cna = sbc.add_mock_copynumber_data(df.copy(), grouping_cols=["species"])
        for c in ("copy_number", "species"):
            assert c in df_cna.columns.to_list()
        assert all(df_cna["copy_number"] >= 0.0)
        assert not any(df_cna["copy_number"].isna())
        for species in df["species"].unique():
            n_unique_vals = len(
                np.unique(df_cna[df_cna.species == species]["copy_number"].values)
            )
            assert n_unique_vals == 1


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
    mod_df = sbc.add_mock_rna_expression_data(df, subgroups=["hugo_symbol"])
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


def test_add_mock_rna_expression_data_grouping_cols():
    df = sbc.generate_mock_achilles_categorical_groups(
        n_genes=10,
        n_sgrnas_per_gene=100,
        n_cell_lines=4,
        n_lineages=2,
        n_batches=1,
        n_screens=1,
    )
    mod_df = sbc.add_mock_rna_expression_data(
        df, grouping_cols=["hugo_symbol", "depmap_id"]
    )
    for col in ("hugo_symbol", "depmap_id", "rna_expr"):
        assert col in mod_df.columns.to_list()
    assert np.all(mod_df["rna_expr"] >= 0.0)
    assert not np.any(mod_df["rna_expr"].isna())
    assert np.all(np.isfinite(mod_df["rna_expr"].values))
    rna_df = mod_df.drop_duplicates(["hugo_symbol", "depmap_id", "rna_expr"])
    gene_cellline_df = mod_df.drop_duplicates(["hugo_symbol", "depmap_id"])
    assert rna_df.shape[0] == gene_cellline_df.shape[0]


@given(st.floats(0, 1, allow_nan=False, allow_infinity=False))
def test_add_mock_is_mutated_data(prob: float):
    df = pd.DataFrame({"A": np.zeros(10000)})
    df = sbc.add_mock_is_mutated_data(df, prob=prob)
    assert df["is_mutated"].mean() == pytest.approx(prob, abs=0.1)


@given(st.floats(0, 1, allow_nan=False, allow_infinity=False))
def test_add_mock_is_mutated_data_grouped(prob: float):
    letters = list(ascii_lowercase)
    grp_a = np.random.choice(letters, size=5, replace=False)
    grp_b = np.random.choice(letters, size=5, replace=False)
    N = 5000
    df = pd.DataFrame(
        {"A": np.random.choice(grp_a, size=N), "B": np.random.choice(grp_b, size=N)}
    )
    df_mut = sbc.add_mock_is_mutated_data(df, grouping_cols=["A", "B"], prob=prob)
    for col in ("A", "B", "is_mutated"):
        assert col in df_mut.columns.tolist()
    assert df_mut["is_mutated"].mean() == pytest.approx(prob, abs=0.1)
    assert (
        df[["A", "B"]].drop_duplicates().shape[0] == df_mut.drop_duplicates().shape[0]
    )


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


@settings(settings.get_profile("slow-adaptive"))
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


@pytest.mark.DEV
@settings(settings.load_profile("slow-adaptive"))
@given(mock_data=generate_data_with_random_params())
def test_sgrnas_uniquely_map_to_genes(mock_data: pd.DataFrame):
    sgrna_gene_map = (
        mock_data[["sgrna", "hugo_symbol"]].drop_duplicates().reset_index(drop=True)
    )
    sgrnas = sgrna_gene_map["sgrna"].values
    assert len(sgrnas) == len(np.unique(sgrnas))


@settings(
    settings.get_profile("slow-adaptive"), suppress_health_check=[HealthCheck.too_slow]
)
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
        # Confirm that each combo happens exactly once.
        assert len(all_sgrnas) == len(cell_line_sgrnas)
        assert len(all_sgrnas.difference(set(cell_line_sgrnas))) == 0


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(mock_data=generate_data_with_random_params())
def test_cn_same_for_gene_cellline_combination(mock_data: pd.DataFrame):
    assert "copy_number" in mock_data.columns.tolist()
    n_genes = len(mock_data.hugo_symbol.unique())
    n_celllines = len(mock_data.depmap_id.unique())
    cn_df = mock_data[["hugo_symbol", "depmap_id", "copy_number"]].drop_duplicates()
    assert cn_df.shape[0] == (n_genes * n_celllines)


#### ---- Test SBC collation ---- ####


# Fixtures and helpers


@pytest.fixture
def centered_eight() -> az.InferenceData:
    x = az.load_arviz_data("centered_eight")
    assert isinstance(x, az.InferenceData)
    return x


@pytest.fixture
def centered_eight_post(centered_eight: az.InferenceData) -> pd.DataFrame:
    x = az.summary(centered_eight)
    assert isinstance(x, pd.DataFrame)
    return x


def return_iris(*args, **kwargs) -> pd.DataFrame:
    return sns.load_dataset("iris")


@pytest.fixture(scope="module")
def simple_model() -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=[1, 2, 3])  # noqa: F841
    return model


@pytest.fixture(scope="module")
def hierarchical_model() -> pm.Model:
    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", 0, 1)
        sigma_alpha = pm.HalfCauchy("sigma_alpha", 1)
        alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, shape=2)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal(  # noqa: F841
            "y", alpha[np.array([0, 0, 1, 1])], sigma, observed=[1, 2, 3, 4]
        )
    return model


# Tests


def test_is_true_value_within_hdi_lower_limit():
    n = 100
    low = pd.Series(list(range(0, n)))
    high = pd.Series([200] * n)
    vals = pd.Series([50] * n)
    is_within = sbc._is_true_value_within_hdi(low, vals, high)
    assert np.all(is_within[:50])
    assert not np.any(is_within[50:])


def test_is_true_value_within_hdi_upper_limit():
    n = 100
    low = pd.Series([0] * n)
    high = pd.Series(list(range(100)))
    vals = pd.Series([50] * n)
    is_within = sbc._is_true_value_within_hdi(low, vals, high)
    assert not np.any(is_within[:51])
    assert np.all(is_within[51:])


def test_get_prior_value_using_index_list_mismatch_index_size():
    a = np.array([4, 3, 2, 1])
    idx: list[int] = []
    with pytest.raises(AssertionError):
        _ = sbc._get_prior_value_using_index_list(a, idx)


def test_get_prior_value_using_index_list_empty_idx():
    a = np.array(4)
    idx: list[int] = []
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_empty_idx_but_not_flat_array():
    a = np.array([4])
    idx: list[int] = []
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_1d():
    a = np.array([4, 3, 2, 1])
    idx = [0]
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4
    idx = [1]
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 3


def test_get_prior_value_using_index_list_2d():
    a = np.arange(9).reshape((3, 3))
    idx = [1, 2]
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == a[1, 2]


@pytest.mark.parametrize(
    "p, res",
    [
        ("a", ["a"]),
        ("abc", ["abc"]),
        ("abc[0]", ["abc", "0"]),
        ("abc[0,2,5]", ["abc", "0", "2", "5"]),
        ("abc[ x, y, z]", ["abc", " x", " y", " z"]),
        ("abc[x,y,z]", ["abc", "x", "y", "z"]),
    ],
)
def test_split_parameter(p: str, res: str):
    assert res == sbc._split_parameter(p)


def test_error_when_incorrect_number_of_results_found(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sbc, "get_posterior_summary_for_file_manager", return_iris)

    n_paths = 20
    fake_paths = [Path(f"fake-path-{i}") for i in range(n_paths)]

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err1:
        sbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths - 1)

    assert err1.value.expected == n_paths - 1
    assert err1.value.found == n_paths

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err2:
        sbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths + 1)

    assert err2.value.expected == n_paths + 1
    assert err2.value.found == n_paths


@pytest.mark.slow
def test_make_priors_dataframe_simple(simple_model: pm.Model):
    with simple_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu", "sigma"]
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu_alpha", "sigma_alpha", "alpha[0]", "alpha[1]", "sigma"]
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical_with_post(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)
        trace = pm.sample(10, tune=10, cores=1, chains=2, return_inferencedata=True)

    parameters: list[str] = az.summary(trace).index.tolist()
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


def test_failure_if_data_does_not_exist(tmp_path: Path):
    with pytest.raises(sbc.SBCResultsNotFoundError):
        sbc.get_posterior_summary_for_file_manager(tmp_path)
