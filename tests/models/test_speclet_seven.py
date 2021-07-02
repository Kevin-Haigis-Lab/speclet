from pathlib import Path
from typing import List

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.misc.test_helpers import generate_model_parameterizations
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_seven import SpecletSeven, SpecletSevenParameterization
from src.project_enums import ModelParameterization as MP

model_parameterizations: List[
    SpecletSevenParameterization
] = generate_model_parameterizations(
    param_class=SpecletSevenParameterization, n_randoms=10
)


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

    top_priors = ["μ_μ_μ_a", "σ_μ_μ_a", "σ_σ_μ_a", "σ_σ_a", "σ"]

    @pytest.mark.slow
    @pytest.mark.parametrize("fit_method", ["mcmc", "advi"])
    def test_model_fitting(self, sp7: SpecletSeven, fit_method: str):
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
            assert sp7.mcmc_results is None
            assert sp7.advi_results is not None

        for p in self.top_priors:
            assert fit_res.posterior[p].shape == (n_chains, n_draws)

    @pytest.mark.parametrize("model_param", model_parameterizations)
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        model_param: SpecletSevenParameterization,
    ):
        sp7 = SpecletSeven(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=CrcDataManager(),
            parameterization=model_param,
        )
        assert sp7.model is None
        sp7.build_model()
        assert sp7.model is not None

        rv_names = pmhelp.get_random_variable_names(sp7.model)
        unobs_names = pmhelp.get_deterministic_variable_names(sp7.model)

        for param_name, param_method in zip(model_param._fields, model_param):
            assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
            assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)
