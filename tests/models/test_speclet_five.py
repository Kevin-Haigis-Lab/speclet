from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.misc import test_helpers as th
from src.models.speclet_five import SpecletFive, SpecletFiveConfiguration


class TestSpecletFive:
    def test_instantiation(self, tmp_path: Path) -> None:
        sp5 = SpecletFive("TEST-MODEL", root_cache_dir=tmp_path)
        assert sp5.model is None

    @pytest.fixture(scope="function")
    def sp5(self, tmp_path: Path) -> SpecletFive:
        sp5 = SpecletFive("TEST-MODEL", root_cache_dir=tmp_path)
        return sp5

    def test_build_model(self, sp5: SpecletFive) -> None:
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletFiveConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        config: SpecletFiveConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def mock_build_model(*args: Any, **kwargs: Any) -> tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletFive, "model_specification", mock_build_model)
        sp5 = SpecletFive(
            "test-model",
            root_cache_dir=tmp_path,
        )
        th.assert_changing_configuration_resets_model(
            sp5, new_config=config, default_config=SpecletFiveConfiguration()
        )

    @pytest.mark.slow
    def test_mcmc_sampling(self, sp5: SpecletFive) -> None:
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None
        assert sp5.observed_var_name is not None
        assert sp5.mcmc_results is None
        _ = sp5.mcmc_sample_model(
            prior_pred_samples=10,
            random_seed=1,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
            },
        )
        assert sp5.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, sp5: SpecletFive) -> None:
        assert sp5.model is None
        sp5.build_model()
        assert sp5.model is not None
        assert sp5.observed_var_name is not None
        assert sp5.advi_results is None
        _ = sp5.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            random_seed=1,
        )
        assert sp5.advi_results is not None

    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletFiveConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        config: SpecletFiveConfiguration,
    ) -> None:
        sp5 = SpecletFive("test-model", root_cache_dir=tmp_path, config=config)
        th.assert_model_reparameterization(sp5, config=config)
