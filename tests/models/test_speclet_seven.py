from pathlib import Path

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_seven import SpecletSeven


@pytest.fixture(autouse=True)
def data_manager_use_test_data(monkeypatch: pytest.MonkeyPatch):
    def mock_get_data_path(*args, **kwargs) -> Path:
        return Path("tests", "depmap_test_data.csv")

    monkeypatch.setattr(CrcDataManager, "get_data_path", mock_get_data_path)


class TestSpecletSeven:
    @pytest.fixture(scope="function")
    def sp7(self, tmp_path: Path) -> SpecletSeven:
        return SpecletSeven("test-model", root_cache_dir=tmp_path)

    def test_init(self, sp7: SpecletSeven):
        assert sp7.model is None
        assert sp7.mcmc_results is None
        assert sp7.advi_results is None
        assert sp7.data_manager is not None

    @pytest.mark.parametrize("noncentered_param", [True, False])
    def test_build_model_spec(self, sp7: SpecletSeven, noncentered_param: bool):
        sp7.noncentered_param = noncentered_param
        sp7.build_model()
        assert sp7.model is not None

    top_priors = ["μ_μ_μ_a", "σ_μ_μ_a", "σ_σ_μ_a", "σ_σ_a", "σ"]

    @pytest.mark.parametrize("fit_method", ["mcmc", "advi"])
    @pytest.mark.parametrize("noncentered_param", [True, False])
    def test_mcmc_sampling(
        self, sp7: SpecletSeven, fit_method: str, noncentered_param: bool
    ):
        sp7.noncentered_param = noncentered_param
        sp7.build_model()
        assert sp7.model is not None

        n_draws, n_chains = 10, 1

        if fit_method == "mcmc":
            fit_res = sp7.mcmc_sample_model(
                mcmc_draws=n_draws,
                tune=10,
                chains=n_chains,
                cores=n_chains,
                target_accept=0.8,
                prior_pred_samples=10,
                post_pred_samples=10,
                ignore_cache=True,
            )
            assert sp7.mcmc_results is not None
            assert sp7.advi_results is None
        else:
            fit_res, _ = sp7.advi_sample_model(
                n_iterations=20,
                draws=n_draws,
                prior_pred_samples=10,
                post_pred_samples=10,
                ignore_cache=True,
            )

        for p in self.top_priors:
            assert fit_res.posterior[p].shape == (n_chains, n_draws)
