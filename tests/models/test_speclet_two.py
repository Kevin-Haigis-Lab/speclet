from pathlib import Path
from typing import Any

import pytest

from src.models.speclet_two import SpecletTwo


def monkey_get_data_path(*args: Any, **kwargs: Any) -> Path:
    return Path("tests", "depmap_test_data.csv")


class TestSpecletTwo:
    @pytest.fixture(scope="function")
    def test_instantiation(self, tmp_path: Path) -> None:
        sp_two = SpecletTwo("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_two.model is None

    @pytest.fixture(scope="function")
    def sp_two(self, tmp_path: Path) -> SpecletTwo:
        sp_two = SpecletTwo("test-model", root_cache_dir=tmp_path, debug=True)
        return sp_two

    def test_build_model(self, sp_two: SpecletTwo) -> None:
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, sp_two: SpecletTwo) -> None:
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None
        assert sp_two.observed_var_name is not None
        assert sp_two.mcmc_results is None
        _ = sp_two.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=1,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp_two.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, sp_two: SpecletTwo) -> None:
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None
        assert sp_two.observed_var_name is not None
        assert sp_two.advi_results is None
        _ = sp_two.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp_two.advi_results is not None
