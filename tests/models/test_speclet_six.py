from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_six import SpecletSix


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


class TestSpecletSix:
    @pytest.fixture(scope="class")
    def data_manager(self) -> CrcDataManager:
        dm = CrcDataManager(debug=True)
        dm.data = (
            dm.get_data()
            .pipe(achelp.subsample_achilles_data, n_genes=5, n_cell_lines=3)
            .pipe(achelp.set_achilles_categorical_columns)
        )
        assert dphelp.nunique(dm.data["hugo_symbol"]) == 5
        assert dphelp.nunique(dm.data["depmap_id"]) == 3
        return dm

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
        model_vars = [v.name for v in sp6.model.free_RVs]
        model_vars = [v.replace("_log__", "") for v in model_vars]
        for expected_v in ["μ_μ_d", "σ_μ_d", "μ_d_offset", "σ_σ_d"]:
            assert expected_v in model_vars

        data_manager.data = None  # clean up changes to data

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
        model_vars = [v.name for v in sp6.model.free_RVs]
        model_vars = [v.replace("_log__", "") for v in model_vars]
        for var in multi_screen_vars:
            assert var not in model_vars

        make_data_multiple_screens(dm=data_manager)
        sp6.build_model()
        model_vars = [v.name for v in sp6.model.free_RVs]
        model_vars = [v.replace("_log__", "") for v in model_vars]
        for var in multi_screen_vars:
            assert var in model_vars

        data_manager.data = None  # clean up changes to data

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
        data_manager.data = None  # clean up changes to data

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
