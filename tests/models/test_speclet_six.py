from pathlib import Path
from string import ascii_letters
from typing import List

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data_processing import achilles as achelp
from src.managers.model_data_managers import CrcDataManager
from src.modeling.pymc3_helpers import get_variable_names
from src.models import speclet_six
from src.models.speclet_six import SpecletSix

chars = [str(i) for i in range(10)] + list(ascii_letters)


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


def make_column_tiled(
    df: pd.DataFrame, df_map: pd.DataFrame, col: str, val: List[str]
) -> pd.DataFrame:
    new_assignemnts = np.tile(val, int(len(df_map) // 2 + 1))
    df_map[col] = new_assignemnts[0 : len(df_map)]
    df_mod = (
        df.drop(columns=[col])
        .merge(df_map, left_index=False, right_index=False)
        .pipe(achelp.set_achilles_categorical_columns)
    )
    return df_mod


def make_data_multiple_lineages(dm: CrcDataManager):
    data = dm.get_data()
    lineage_map = achelp.make_cell_line_to_lineage_mapping_df(data)
    new_lineages = ["lineage_A", "lineage_B"]
    dm.data = make_column_tiled(data, lineage_map, "lineage", new_lineages)


def make_data_multiple_screens(dm: CrcDataManager):
    data = dm.get_data()
    batch_map = achelp.data_batch_indices(data).batch_to_screen_map
    new_screens = ["screen_A", "screen_B"]
    dm.data = make_column_tiled(data, batch_map, "screen", new_screens)


@st.composite
def copynumber_dataframe(draw, group_name: str) -> pd.DataFrame:
    groups = draw(st.lists(st.text(alphabet=chars), min_size=1))
    values = [
        draw(
            st.lists(
                st.floats(
                    min_value=-1000.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
            )
        )
        for _ in groups
    ]
    return (
        pd.DataFrame({group_name: groups, "copy_number": values})
        .explode("copy_number")
        .astype({"copy_number": float})
        .reset_index(drop=True)
    )


class TestSpecletSix:
    @pytest.fixture(scope="function")
    def data_manager(self, monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(debug=True)
        return dm

    @settings(deadline=None)
    @given(copynumber_dataframe(group_name="depmap_id"))
    def test_centered_copynumber_by_cellline(self, df: pd.DataFrame):
        mod_df = speclet_six.centered_copynumber_by_cellline(df.copy())
        for cell_line in mod_df["depmap_id"].unique():
            avg = mod_df.query(f"depmap_id == '{cell_line}'")[
                "copy_number_cellline"
            ].mean()
            assert avg == pytest.approx(0.0, abs=0.001)

    @settings(deadline=None)
    @given(copynumber_dataframe(group_name="hugo_symbol"))
    def test_centered_copynumber_by_gene(self, df: pd.DataFrame):
        mod_df = speclet_six.centered_copynumber_by_gene(df.copy())
        for gene in mod_df["hugo_symbol"].unique():
            avg = mod_df.query(f"hugo_symbol == '{gene}'")["copy_number_gene"].mean()
            assert avg == pytest.approx(0.0, abs=0.001)

    @given(st.lists(st.booleans(), max_size=100))
    def test_converting_is_mutated_column(self, is_mutated: List[bool]):
        df = pd.DataFrame({"is_mutated": is_mutated})
        mod_df = speclet_six.convert_is_mutated_to_numeric(df)
        assert mod_df["is_mutated"].dtype == np.int64

    def test_instantiation(self, tmp_path: Path):
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None

    def test_model_with_multiple_cell_line_lineages(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        make_data_multiple_lineages(data_manager)
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        sp6.build_model()
        assert sp6.model is not None
        model_vars = get_variable_names(sp6.model, rm_log=True)
        for expected_v in ["μ_μ_d", "σ_μ_d", "μ_d_offset", "σ_σ_d"]:
            assert expected_v in model_vars

    def test_model_with_multiple_screens(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        d = data_manager.get_data().copy()
        d["screen"] = "screen_A"
        d = achelp.set_achilles_categorical_columns(d)
        data_manager.data = d

        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )

        multi_screen_vars = ["μ_μ_j", "σ_μ_j", "σ_σ_j", "μ_j_offset"]

        sp6.build_model()
        assert sp6.model is not None
        model_vars = get_variable_names(sp6.model, rm_log=True)
        for var in multi_screen_vars:
            assert var not in model_vars

        make_data_multiple_screens(dm=data_manager)
        sp6.build_model()
        assert sp6.model is not None
        model_vars = get_variable_names(sp6.model, rm_log=True)
        for var in multi_screen_vars:
            assert var in model_vars

    def test_model_with_optional_cellline_cn_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )

        expected_vars = set(["μ_k", "σ_k", "k_offset", "k"])

        for with_cell_line_cov in [False, True]:
            sp6.cell_line_cna_cov = with_cell_line_cov
            assert sp6.model is None
            sp6.build_model()
            assert sp6.model is not None
            model_vars = get_variable_names(sp6.model, rm_log=True)
            for v in expected_vars:
                assert (v in model_vars) == with_cell_line_cov

    def test_model_with_optional_gene_cn_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )

        expected_vars = set(["μ_n", "σ_n", "n_offset", "n"])

        for with_gene_cov in [False, True]:
            sp6.gene_cna_cov = with_gene_cov
            assert sp6.model is None
            sp6.build_model()
            assert sp6.model is not None
            model_vars = get_variable_names(sp6.model, rm_log=True)
            for v in expected_vars:
                assert (v in model_vars) == with_gene_cov

    def test_model_with_optional_gene_lineage_rna_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )

        expected_vars = set(["μ_q", "σ_q", "q_offset", "q"])

        for with_rna_cov in [False, True]:
            sp6.rna_cov = with_rna_cov
            assert sp6.model is None
            sp6.build_model()
            assert sp6.model is not None
            model_vars = get_variable_names(sp6.model, rm_log=True)
            for v in expected_vars:
                assert (v in model_vars) == with_rna_cov

    def test_model_with_optional_gene_lineage_mutation_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )

        expected_vars = set(["μ_m", "σ_m", "m_offset", "m"])

        for with_mut_cov in [False, True]:
            sp6.mutation_cov = with_mut_cov
            assert sp6.model is None
            sp6.build_model()
            assert sp6.model is not None
            model_vars = get_variable_names(sp6.model, rm_log=True)
            for v in expected_vars:
                assert (v in model_vars) == with_mut_cov

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_multiple_lineages(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        make_data_multiple_lineages(data_manager)
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_with_optional_covariates(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            cell_line_cna_cov=True,
            gene_cna_cov=True,
            rna_cov=True,
            mutation_cov=True,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.advi_results is None
        _ = sp6.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.advi_results is not None
