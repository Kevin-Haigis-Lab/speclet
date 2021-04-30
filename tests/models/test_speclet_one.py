from pathlib import Path

import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.models.speclet_one import SpecletOne


class TestSpecletOne:
    def test_instantiation(self, tmp_path: Path):
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_one.model is None

    def test_build_model(self, tmp_path: Path):
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path, debug=True)
        d = sp_one.data_manager.get_data()
        sp_one.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(sp_one.data_manager.data["hugo_symbol"]) == 5
        assert dphelp.nunique(sp_one.data_manager.data["depmap_id"]) == 3
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path):
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path, debug=True)
        d = sp_one.data_manager.get_data()
        sp_one.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(sp_one.data_manager.data["hugo_symbol"]) == 5
        assert dphelp.nunique(sp_one.data_manager.data["depmap_id"]) == 3
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None
        assert sp_one.observed_var_name is not None
        assert sp_one.mcmc_results is None
        _ = sp_one.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_one.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path):
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path, debug=True)
        d = sp_one.data_manager.get_data()
        sp_one.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(sp_one.data_manager.data["hugo_symbol"]) == 5
        assert dphelp.nunique(sp_one.data_manager.data["depmap_id"]) == 3
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None
        assert sp_one.observed_var_name is not None
        assert sp_one.advi_results is None
        _ = sp_one.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_one.advi_results is not None
