from pathlib import Path
from typing import Any, Tuple

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.managers.model_data_managers import CrcDataManager
from src.misc import test_helpers as th
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_four import SpecletFour, SpecletFourConfiguration


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
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp4.mcmc_results is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletFourConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletFourConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def mock_build_model(*args, **kwargs) -> Tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletFour, "model_specification", mock_build_model)
        sp4 = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        th.assert_changing_configuration_resets_model(
            sp4, new_config=config, default_config=SpecletFourConfiguration()
        )

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

    @settings(
        settings.get_profile("slow-adaptive"),
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

        def pre_check_callback(param_name: str, *args: Any, **kwargs: Any) -> bool:
            if param_name == "β" and not config.copy_number_cov:
                return True
            elif param_name == "copy_number_cov":
                return True
            return False

        th.assert_model_reparameterization(
            sp4, config=config, pre_check_callback=pre_check_callback
        )
