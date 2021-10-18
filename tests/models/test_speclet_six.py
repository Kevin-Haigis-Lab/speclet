from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.data_processing import achilles as achelp
from src.misc import test_helpers as th
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_model import SpecletModelDataManager
from src.models.speclet_six import SpecletSix, SpecletSixConfiguration


def make_column_tiled(
    df: pd.DataFrame, df_map: pd.DataFrame, col: str, val: list[str]
) -> pd.DataFrame:
    new_assignemnts = np.tile(val, int(len(df_map) // 2 + 1))
    df_map[col] = new_assignemnts[0 : len(df_map)]
    df_mod = (
        df.drop(columns=[col])
        .merge(df_map, left_index=False, right_index=False)
        .pipe(achelp.set_achilles_categorical_columns)
    )
    return df_mod


def make_data_multiple_lineages(dm: SpecletModelDataManager) -> None:
    data = dm.get_data()
    lineage_map = achelp.make_cell_line_to_lineage_mapping_df(data)
    new_lineages = ["lineage_A", "lineage_B"]
    dm.set_data(
        make_column_tiled(data, lineage_map, "lineage", new_lineages),
        apply_transformations=False,
    )


def make_data_multiple_screens(dm: SpecletModelDataManager) -> None:
    data = dm.get_data()
    batch_map = achelp.data_batch_indices(data).batch_to_screen_map
    new_screens = ["screen_A", "screen_B"]
    dm.set_data(
        make_column_tiled(data, batch_map, "screen", new_screens),
        apply_transformations=False,
    )


class TestSpecletSix:
    def test_instantiation(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None

    def test_build_model(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletSixConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        config: SpecletSixConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def mock_build_model(*args: Any, **kwargs: Any) -> tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletSix, "model_specification", mock_build_model)
        sp6 = SpecletSix("test-model", root_cache_dir=tmp_path, debug=True)
        th.assert_changing_configuration_resets_model(
            sp6, new_config=config, default_config=SpecletSixConfiguration()
        )

    def test_model_with_multiple_cell_line_lineages(self, tmp_path: Path) -> None:
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
        )
        make_data_multiple_lineages(sp6.data_manager)
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for expected_v in ["μ_μ_d", "σ_μ_d", "μ_d", "σ_σ_d"]:
            assert expected_v in model_vars

    def test_model_with_multiple_screens(
        self,
        tmp_path: Path,
    ) -> None:
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
        )

        d = sp6.data_manager.get_data().copy()
        d["screen"] = "screen_A"
        d = achelp.set_achilles_categorical_columns(d)
        sp6.data_manager.set_data(d)

        multi_screen_vars = ["μ_μ_j", "σ_μ_j", "σ_σ_j"]

        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        print(model_vars)
        for var in multi_screen_vars:
            assert var not in model_vars

        make_data_multiple_screens(dm=sp6.data_manager)
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for var in multi_screen_vars:
            assert var in model_vars

    def test_transformed_data_has_expected_shape(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL_fsejio", root_cache_dir=tmp_path)
        data = sp6.data_manager.get_data()
        for _ in range(3):
            data = sp6.data_manager.get_data()
        assert data.copy_number_cellline.values.ndim == 1
        assert data.copy_number_gene.values.ndim == 1

    @pytest.mark.parametrize(
        "arg_name, expected_vars",
        [
            ("gene_cna_cov", {"μ_n", "σ_n", "n"}),
            ("rna_cov", {"μ_q", "σ_q", "q"}),
            ("mutation_cov", {"μ_m", "σ_m", "m"}),
            ("cell_line_cna_cov", {"μ_k", "σ_k", "k"}),
        ],
    )
    @pytest.mark.parametrize("arg_value", [True, False])
    def test_model_with_optional_covariates(
        self, tmp_path: Path, arg_name: str, expected_vars: set[str], arg_value: bool
    ) -> None:
        config = SpecletSixConfiguration(**{arg_name: arg_value})
        cache_dir = tmp_path / arg_name
        cache_dir.mkdir()
        sp6 = SpecletSix(
            f"TEST-MODEL_{arg_name}",
            root_cache_dir=cache_dir,
            debug=True,
            config=config,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for v in expected_vars:
            assert (v in model_vars) == arg_value

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            prior_pred_samples=10,
            random_seed=1,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
            },
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_multiple_lineages(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        make_data_multiple_lineages(sp6.data_manager)
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            prior_pred_samples=10,
            random_seed=1,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
            },
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_with_optional_covariates(self, tmp_path: Path) -> None:
        config = SpecletSixConfiguration(
            **{
                "cell_line_cna_cov": True,
                "gene_cna_cov": True,
                "rna_cov": True,
                "mutation_cov": True,
            }
        )
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, config=config
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            prior_pred_samples=10,
            random_seed=1,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
            },
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path) -> None:
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.advi_results is None
        _ = sp6.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp6.advi_results is not None

    @settings(
        settings.get_profile("slow-adaptive"),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletSixConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        config: SpecletSixConfiguration,
    ) -> None:
        sp6 = SpecletSix(
            "test-model", root_cache_dir=tmp_path, debug=True, config=config
        )

        optional_param_to_name: dict[str, str] = {
            "k": "cell_line_cna_cov",
            "n": "gene_cna_cov",
            "q": "rna_cov",
            "m": "mutation_cov",
        }

        def pre_check_callback(param_name: str, *args: Any, **kwargs: Any) -> bool:
            if len(param_name) > 1:
                return True
            if (
                param_name in optional_param_to_name.keys()
                and not config.dict()[optional_param_to_name[param_name]]
            ):
                return True
            return False

        th.assert_model_reparameterization(
            sp6, config=config, pre_check_callback=pre_check_callback
        )
