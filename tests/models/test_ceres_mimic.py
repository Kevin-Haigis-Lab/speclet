#!/usr/bin/env python3

from pathlib import Path

import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.models.ceres_mimic import CeresMimic


class TestCeresMimic:
    def test_instantiation(self, tmp_path: Path):
        cm = CeresMimic("test-model", root_cache_dir=tmp_path, debug=True)
        assert cm.model is None

    def build_model(self, cm: CeresMimic):
        d = cm.data_manager.get_data()
        cm.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(cm.data_manager.data["hugo_symbol"] == 5)
        assert dphelp.nunique(cm.data_manager.data["depmap_id"] == 3)
        assert cm.model is None
        cm.build_model()
        assert cm.model is not None

    def test_build_model_base(self, tmp_path: Path):
        cm = CeresMimic("test-model", root_cache_dir=tmp_path, debug=True)
        self.build_model(cm)

    def test_build_model_copynumber_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, copynumber_cov=True
        )
        self.build_model(cm)

    def test_build_model_sgrna_intercept_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, sgrna_intercept_cov=True
        )
        self.build_model(cm)

    def mcmc_sampling(self, cm: CeresMimic):
        d = cm.data_manager.get_data()
        cm.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(cm.data_manager.data["hugo_symbol"] == 5)
        assert dphelp.nunique(cm.data_manager.data["depmap_id"] == 3)
        assert cm.model is None
        cm.build_model()
        assert cm.model is not None
        assert cm.observed_var_name is not None
        assert cm.mcmc_results is None
        _ = cm.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert cm.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_base(self, tmp_path: Path):
        cm = CeresMimic("test-model", root_cache_dir=tmp_path, debug=True)
        self.mcmc_sampling(cm)
        assert "β" not in list(cm.mcmc_results.posterior.keys())
        assert "o" not in list(cm.mcmc_results.posterior.keys())

    @pytest.mark.slow
    def test_mcmc_sampling_copynumber_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, copynumber_cov=True
        )
        self.mcmc_sampling(cm)
        assert "β" in list(cm.mcmc_results.posterior.keys())
        assert "o" not in list(cm.mcmc_results.posterior.keys())

    @pytest.mark.slow
    def test_mcmc_sampling_sgrna_intercept_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, sgrna_intercept_cov=True
        )
        self.mcmc_sampling(cm)
        assert "β" not in list(cm.mcmc_results.posterior.keys())
        assert "o" in list(cm.mcmc_results.posterior.keys())

    def advi_sampling(self, cm: CeresMimic):
        d = cm.data_manager.get_data()
        cm.data_manager.data = achelp.subsample_achilles_data(
            d, n_genes=5, n_cell_lines=3
        ).pipe(achelp.set_achilles_categorical_columns)
        assert dphelp.nunique(cm.data_manager.data["hugo_symbol"] == 5)
        assert dphelp.nunique(cm.data_manager.data["depmap_id"] == 3)
        assert cm.model is None
        cm.build_model()
        assert cm.model is not None
        assert cm.observed_var_name is not None
        assert cm.advi_results is None
        _ = cm.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert cm.advi_results is not None

    @pytest.mark.slow
    def test_advi_sampling_base(self, tmp_path: Path):
        cm = CeresMimic("test-model", root_cache_dir=tmp_path, debug=True)
        self.advi_sampling(cm)
        assert "β" not in list(cm.advi_results[0].posterior.keys())
        assert "o" not in list(cm.advi_results[0].posterior.keys())

    @pytest.mark.slow
    def test_advi_sampling_copynumber_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, copynumber_cov=True
        )
        self.advi_sampling(cm)
        assert "β" in list(cm.advi_results[0].posterior.keys())
        assert "o" not in list(cm.advi_results[0].posterior.keys())

    @pytest.mark.slow
    def test_advi_sampling_sgrna_intercept_cov(self, tmp_path: Path):
        cm = CeresMimic(
            "test-model", root_cache_dir=tmp_path, debug=True, sgrna_intercept_cov=True
        )
        self.advi_sampling(cm)
        assert "β" not in list(cm.advi_results[0].posterior.keys())
        assert "o" in list(cm.advi_results[0].posterior.keys())
