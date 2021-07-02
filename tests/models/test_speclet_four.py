from pathlib import Path
from typing import List

import pytest

from src.managers.model_data_managers import CrcDataManager
from src.misc.test_helpers import generate_model_parameterizations
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_four import SpecletFour, SpecletFourParameterization
from src.project_enums import ModelParameterization as MP


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


model_parameterizations: List[
    SpecletFourParameterization
] = generate_model_parameterizations(
    param_class=SpecletFourParameterization, n_randoms=10
)


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

    @pytest.mark.parametrize("copy_cov", [True, False])
    def test_switching_copynumber_covariate(
        self, tmp_path: Path, data_manager: CrcDataManager, copy_cov: bool
    ):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            copy_number_cov=copy_cov,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        rv_names += [v.name for v in sp_four.model.unobserved_RVs]
        assert ("β" in set(rv_names)) == copy_cov

        sp_four.copy_number_cov = not copy_cov
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        rv_names = [v.name for v in sp_four.model.free_RVs]
        rv_names += [v.name for v in sp_four.model.unobserved_RVs]
        assert ("β" in set(rv_names)) != copy_cov

    @pytest.mark.parametrize("copy_cov", [True, False])
    @pytest.mark.parametrize("model_param", model_parameterizations)
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        data_manager: CrcDataManager,
        copy_cov: bool,
        model_param: SpecletFourParameterization,
    ):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            copy_number_cov=copy_cov,
            parameterization=model_param,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None

        rv_names = pmhelp.get_random_variable_names(sp_four.model)
        unobs_names = pmhelp.get_deterministic_variable_names(sp_four.model)
        all_var_names = rv_names + unobs_names

        assert ("β" in set(all_var_names)) == copy_cov

        for param_name, param_method in zip(model_param._fields, model_param):
            if param_name == "β":
                continue
            assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
            assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)
