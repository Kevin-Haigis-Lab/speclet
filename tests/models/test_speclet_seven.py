from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.misc import test_helpers as th
from src.modeling import pymc3_helpers as pmhelp
from src.models.speclet_seven import SpecletSeven, SpecletSevenConfiguration


class TestSpecletSeven:
    @pytest.fixture(scope="function")
    def sp7(
        self,
        tmp_path: Path,
    ) -> SpecletSeven:
        return SpecletSeven("test-model", root_cache_dir=tmp_path, debug=True)

    def test_init(self, sp7: SpecletSeven) -> None:
        assert sp7.model is None
        assert sp7.mcmc_results is None
        assert sp7.advi_results is None
        assert sp7.data_manager is not None

    def test_build_model(self, sp7: SpecletSeven) -> None:
        assert sp7.model is None
        sp7.build_model()
        assert sp7.model is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("fit_method", ["mcmc", "advi"])
    def test_model_fitting(self, sp7: SpecletSeven, fit_method: str) -> None:
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
                ignore_cache=True,
            )
            assert sp7.mcmc_results is not None
            assert sp7.advi_results is None
        else:
            _, _ = sp7.advi_sample_model(
                n_iterations=20,
                draws=n_draws,
                prior_pred_samples=10,
                ignore_cache=True,
            )
            assert sp7.mcmc_results is None
            assert sp7.advi_results is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletSevenConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        config: SpecletSevenConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def mock_build_model(*args: Any, **kwargs: Any) -> tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletSeven, "model_specification", mock_build_model)
        sp7 = SpecletSeven("test-model", root_cache_dir=tmp_path, debug=True)
        th.assert_changing_configuration_resets_model(
            sp7, new_config=config, default_config=SpecletSevenConfiguration()
        )

    @settings(
        settings.get_profile("slow-adaptive"),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletSevenConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        config: SpecletSevenConfiguration,
    ) -> None:
        sp7 = SpecletSeven(
            "test-model", root_cache_dir=tmp_path, debug=True, config=config
        )

        optional_param_to_name: dict[str, str] = {
            "k": "cell_line_cna_cov",
            "n": "gene_cna_cov",
            "q": "rna_cov",
            "m": "mutation_cov",
            "j": "batch_cov",
        }

        def pre_check_callback(param_name: str, *args: Any, **kwargs: Any) -> bool:
            if len(param_name) > 1:
                return True
            if (
                param_name in optional_param_to_name.keys()
                and not config.dict()[optional_param_to_name[param_name]]
            ):
                return True
            return False

        th.assert_model_reparameterization(
            sp7, config=config, pre_check_callback=pre_check_callback
        )

        assert sp7.model is not None
        all_vars = pmhelp.get_variable_names(sp7.model)
        if config.batch_cov:
            assert "j" in all_vars
        if config.cell_line_cna_cov:
            assert "k" in all_vars
        if config.gene_cna_cov:
            assert "n" in all_vars
        if config.rna_cov:
            assert "q" in all_vars
        if config.mutation_cov:
            assert "m" in all_vars
