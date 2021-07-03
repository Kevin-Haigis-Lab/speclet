from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.managers.model_data_managers import CrcDataManager
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_four import SpecletFour, SpecletFourConfiguration
from src.project_enums import ModelParameterization as MP


class TestSpecletFour:
    def test_instantiation(self, tmp_path: Path):
        sp4 = SpecletFour("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp4.model is None

    def test_build_model(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp4 = SpecletFour(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )
        assert sp4.model is None
        sp4.build_model()
        assert sp4.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp4 = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        assert sp4.model is None
        sp4.build_model()
        assert sp4.model is not None
        assert sp4.observed_var_name is not None
        assert sp4.mcmc_results is None
        _ = sp4.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp4.mcmc_results is not None

    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletFourConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletFourConfiguration,
    ):
        sp4 = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        assert sp4.model is None
        sp4.build_model()
        assert sp4.model is not None
        sp4.set_config(config.dict())
        if config == SpecletFourConfiguration():
            assert sp4.model is not None
        else:
            assert sp4.model is None

    @pytest.mark.parametrize("copy_cov", [True, False])
    def test_switching_copynumber_covariate(
        self, tmp_path: Path, mock_crc_dm: CrcDataManager, copy_cov: bool
    ):
        sp4 = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        _config = sp4.config.copy()
        _config.copy_number_cov = copy_cov
        sp4.set_config(_config.dict())
        assert sp4.model is None
        sp4.build_model()
        assert sp4.model is not None
        var_names = pmhelp.get_variable_names(sp4.model)
        assert ("β" in set(var_names)) == copy_cov

    @pytest.mark.DEV
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletFourConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletFourConfiguration,
    ):
        sp4 = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=config,
        )
        assert sp4.model is None
        sp4.build_model()
        assert sp4.model is not None

        rv_names = pmhelp.get_random_variable_names(sp4.model)
        unobs_names = pmhelp.get_deterministic_variable_names(sp4.model)
        all_var_names = rv_names + unobs_names

        assert ("β" in set(all_var_names)) == config.copy_number_cov

        for param_name, param_method in config.dict().items():
            if param_name == "β" and not config.copy_number_cov:
                continue
            elif param_name == "copy_number_cov":
                continue
            assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
            assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)
