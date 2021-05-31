from pathlib import Path

import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_five import SpecletFive


class TestSpecletFive:
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
        sp5 = SpecletFive("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp5.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletFive(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletFive(
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
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp5 = SpecletFive(
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
