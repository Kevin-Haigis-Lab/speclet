from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.managers.model_data_managers import CrcDataManager
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_five import SpecletFive, SpecletFiveConfiguration
from src.project_enums import ModelParameterization as MP


class TestSpecletFive:
    def test_instantiation(self, tmp_path: Path):
        sp5 = SpecletFive("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp5.model is None

    def test_build_model(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp5 = SpecletFive(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp5 = SpecletFive(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
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
    def test_advi_sampling(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp5 = SpecletFive(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
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
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletFiveConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletFiveConfiguration,
    ):
        sp5 = SpecletFive(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=config,
        )
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None

        rv_names = pmhelp.get_random_variable_names(sp5.model)
        unobs_names = pmhelp.get_deterministic_variable_names(sp5.model)
        for param_name, param_method in config.dict().items():
            assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
            assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)
