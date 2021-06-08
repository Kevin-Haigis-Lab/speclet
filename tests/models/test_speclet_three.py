from pathlib import Path

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_three import SpecletThree


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


class TestSpecletThree:
    @pytest.fixture(scope="function")
    def data_manager(self, monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(debug=True)
        return dm

    def test_instantiation(self, tmp_path: Path):
        sp3 = SpecletThree("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp3.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp3 = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp3.model is None
        sp3.build_model()
        assert sp3.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp3 = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp3.model is None
        sp3.build_model()
        assert sp3.model is not None
        assert sp3.observed_var_name is not None
        assert sp3.mcmc_results is None
        _ = sp3.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp3.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp3 = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp3.model is None
        sp3.build_model()
        assert sp3.model is not None
        assert sp3.observed_var_name is not None
        assert sp3.advi_results is None
        _ = sp3.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp3.advi_results is not None

    def test_switching_parameterization(self, tmp_path: Path):
        dm = CrcDataManager(debug=True)
        dm.data = dm.generate_mock_data("small")  # Use mock data.
        sp3 = SpecletThree(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=dm,
            noncentered_param=False,
        )
        sp3.build_model()
        assert sp3.model is not None
        rv_names = [v.name for v in sp3.model.free_RVs]
        assert not any(["offset" in name for name in rv_names])
        sp3.noncentered_param = True
        assert sp3.model is None
        sp3.build_model()
        assert sp3.model is not None
        rv_names = [v.name for v in sp3.model.free_RVs]
        assert any(["offset" in name for name in rv_names])
