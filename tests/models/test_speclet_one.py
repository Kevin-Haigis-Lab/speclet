from pathlib import Path
from typing import Any

import pytest

from src.models.speclet_one import SpecletOne


def monkey_get_data_path(*args: Any, **kwargs: Any) -> Path:
    return Path("tests", "depmap_test_data.csv")


class TestSpecletOne:
    def test_instantiation(self, tmp_path: Path) -> None:
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path)
        assert sp_one.model is None

    @pytest.fixture(scope="function")
    def sp_one(self, tmp_path: Path) -> SpecletOne:
        return SpecletOne("test-model", root_cache_dir=tmp_path)

    def test_build_model(self, sp_one: SpecletOne) -> None:
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, sp_one: SpecletOne) -> None:
        sp_one.build_model()
        assert sp_one.model is not None
        assert sp_one.observed_var_name is not None
        assert sp_one.mcmc_results is None
        _ = sp_one.mcmc_sample_model(
            prior_pred_samples=10,
            random_seed=1,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
            },
        )
        assert sp_one.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, sp_one: SpecletOne) -> None:
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None
        assert sp_one.observed_var_name is not None
        assert sp_one.advi_results is None
        _ = sp_one.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp_one.advi_results is not None
