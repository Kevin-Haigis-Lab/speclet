from pathlib import Path

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_one import SpecletOne


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


class TestSpecletOne:
    @pytest.fixture(scope="function")
    def data_manager(self, monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(debug=True)
        return dm

    def test_instantiation(self, tmp_path: Path):
        sp_one = SpecletOne("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_one.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_one = SpecletOne(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_one.model is None
        sp_one.build_model()
        assert sp_one.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_one = SpecletOne(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        sp_one.build_model()
        assert sp_one.model is not None
        assert sp_one.observed_var_name is not None
        assert sp_one.mcmc_results is None
        _ = sp_one.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp_one.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_one = SpecletOne(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
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
