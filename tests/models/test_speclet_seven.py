from pathlib import Path
from typing import Tuple

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.managers.model_data_managers import CrcDataManager
from src.misc import test_helpers as th
from src.models.speclet_seven import SpecletSeven, SpecletSevenConfiguration


@pytest.fixture(autouse=True)
def data_manager_use_test_data(monkeypatch: pytest.MonkeyPatch):
    def mock_get_data_path(*args, **kwargs) -> Path:
        return Path("tests", "depmap_test_data.csv")

    monkeypatch.setattr(CrcDataManager, "get_data_path", mock_get_data_path)


class TestSpecletSeven:
    @pytest.fixture(scope="function")
    def sp7(self, tmp_path: Path) -> SpecletSeven:
        return SpecletSeven("test-model", root_cache_dir=tmp_path)

    def test_init(self, sp7: SpecletSeven):
        assert sp7.model is None
        assert sp7.mcmc_results is None
        assert sp7.advi_results is None
        assert sp7.data_manager is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("fit_method", ["mcmc", "advi"])
    def test_model_fitting(self, sp7: SpecletSeven, fit_method: str):
        sp7.build_model()
        assert sp7.model is not None

        n_draws, n_chains = 10, 1

        if fit_method == "mcmc":
            _ = sp7.mcmc_sample_model(
                draws=n_draws,
                tune=10,
                chains=n_chains,
                cores=n_chains,
                target_accept=0.8,
                prior_pred_samples=10,
                post_pred_samples=10,
                ignore_cache=True,
            )
            assert sp7.mcmc_results is not None
            assert sp7.advi_results is None
        else:
            _, _ = sp7.advi_sample_model(
                n_iterations=20,
                draws=n_draws,
                prior_pred_samples=10,
                post_pred_samples=10,
                ignore_cache=True,
            )
            assert sp7.mcmc_results is None
            assert sp7.advi_results is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletSevenConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletSevenConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def mock_build_model(*args, **kwargs) -> Tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletSeven, "model_specification", mock_build_model)
        sp7 = SpecletSeven(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        # th.assert_changing_configuration_resets_model(
        #     sp7, new_config=config, default_config=SpecletSevenConfiguration()
        # )

    @settings(
        settings.get_profile("slow-adaptive"),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletSevenConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletSevenConfiguration,
    ):
        sp7 = SpecletSeven(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=config,
        )
        # th.assert_model_reparameterization(sp7, config=config)
