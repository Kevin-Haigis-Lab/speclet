from pathlib import Path

import numpy as np
import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_six import SpecletSix


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

    def make_data_multiple_lineages(self, dm: CrcDataManager):
        data = dm.get_data()
        lineage_map = achelp.make_cell_line_to_lineage_mapping_df(data)
        assert len(lineage_map) > 1
        new_lineages = ["lineageA", "lineageB"]
        lineage_assignments = np.tile(new_lineages, int(len(lineage_map) // 2 + 1))
        lineage_map["lineage"] = lineage_assignments[0 : len(lineage_map)]
        dm.data = (
            data.drop(columns=["lineage"])
            .merge(lineage_map, on="depmap_id")
            .pipe(achelp.set_achilles_categorical_columns)
        )
        new_lineage_map = achelp.make_cell_line_to_lineage_mapping_df(dm.get_data())
        assert len(new_lineage_map.lineage.unique()) > 1

    def test_instantiation(self, tmp_path: Path):
        sp5 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp5.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None

    def test_model_with_multiple_cell_line_lineages(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        self.make_data_multiple_lineages(data_manager)
        sp5 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        sp5.build_model()
        model_vars = [v.name for v in sp5.model.free_RVs]
        model_vars = [v.replace("_log__", "") for v in model_vars]
        for expected_v in ["μ_μ_d", "σ_μ_d", "μ_d_offset", "σ_σ_d"]:
            assert expected_v in model_vars

        data_manager.data = None  # clean up changes to data

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None
        assert sp5.observed_var_name is not None
        assert sp5.mcmc_results is None
        _ = sp5.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp5.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_multiple_lineages(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        self.make_data_multiple_lineages(data_manager)
        sp5 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None
        assert sp5.observed_var_name is not None
        assert sp5.mcmc_results is None
        _ = sp5.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp5.mcmc_results is not None
        data_manager.data = None  # clean up changes to data

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None
        assert sp5.observed_var_name is not None
        assert sp5.advi_results is None
        _ = sp5.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp5.advi_results is not None
