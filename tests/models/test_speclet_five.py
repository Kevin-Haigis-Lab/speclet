from pathlib import Path
from random import choices
from typing import List, Tuple

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_five import SpecletFive, SpecletFiveParameterization
from src.project_enums import ModelParameterization as MP


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


_params: List[MP] = [MP.CENTERED, MP.NONCENTERED]
_random_MP_idx: List[Tuple[int, ...]] = [tuple(choices((0, 1), k=4)) for _ in range(6)]
_random_MP_idx = list(set(_random_MP_idx))

model_parameterizations: List[SpecletFiveParameterization] = [
    SpecletFiveParameterization(_params[h], _params[d], _params[beta], _params[eta])
    for h, d, beta, eta in _random_MP_idx
]


class TestSpecletFive:
    @pytest.fixture(scope="function")
    def data_manager(self, monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
        monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
        dm = CrcDataManager(debug=True)
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

    @pytest.mark.DEV
    @pytest.mark.parametrize("model_param", model_parameterizations)
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        data_manager: CrcDataManager,
        model_param: SpecletFiveParameterization,
    ):
        sp_four = SpecletFive(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            parameterization=model_param,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None

        rv_names = pmhelp.get_random_variable_names(sp_four.model)
        unobs_names = pmhelp.get_deterministic_variable_names(sp_four.model)

        for param_name, param_method in zip(model_param._fields, model_param):
            assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
            assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)
