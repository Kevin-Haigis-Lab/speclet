from pathlib import Path

import numpy as np
import pytest

from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_four import SpecletFour


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


class TestSpecletFour:
    @pytest.fixture(scope="function")
    def data_manager(self, monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(debug=True)
        return dm

    def test_instantiation(self, tmp_path: Path):
        sp_four = SpecletFour("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_four.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_four = SpecletFour(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        assert sp_four.observed_var_name is not None
        assert sp_four.mcmc_results is None
        _ = sp_four.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_four.mcmc_results is not None

    def test_switching_copynumber_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            copy_number_cov=False,
            noncentered_param=False,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        assert np.sum(np.array(rv_names) == "β") == 0

        sp_four.copy_number_cov = True
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        assert any([v == "β" for v in rv_names])

        sp_four.noncentered_param = True
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        assert any([v == "β_offset" for v in rv_names])

    def test_switching_noncentered_parameterization(
        self, tmp_path: Path, data_manager: CrcDataManager
    ):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            noncentered_param=False,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        assert not any(["offset" in n for n in rv_names])

        sp_four.noncentered_param = True
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        assert any(["offset" in n for n in rv_names])
