from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.modeling.pymc3_helpers import get_variable_names
from src.models.ceres_mimic import CeresMimic, CeresMimicConfiguration


class TestCeresMimic:
    def test_instantiation(self, tmp_path: Path) -> None:
        cm = CeresMimic("test-model", root_cache_dir=tmp_path, debug=True)
        assert cm.model is None

    @settings(
        settings.load_profile("slow-adaptive"),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(CeresMimicConfiguration))
    def test_configuration(
        self,
        config: CeresMimicConfiguration,
        tmp_path: Path,
    ) -> None:
        cm = CeresMimic(
            "test-model",
            root_cache_dir=tmp_path,
            config=config,
        )

        assert cm.model is None
        cm.build_model()
        assert cm.model is not None

        assert cm.copynumber_cov == config.copynumber_cov
        assert cm.sgrna_intercept_cov == config.sgrna_intercept_cov
        model_params = get_variable_names(cm.model)

        def assert_maybe_all_in(
            expected_vars: list[str], maybe: bool, all_params: list[str]
        ) -> None:
            assert all([(v in all_params) == maybe for v in expected_vars])

        exp_cn_vars = ["μ_β", "σ_β", "β"]
        exp_sgrna_vars = ["μ_og", "σ_og", "μ_o", "σ_o", "o"]
        assert_maybe_all_in(exp_cn_vars, config.copynumber_cov, model_params)
        assert_maybe_all_in(exp_sgrna_vars, config.sgrna_intercept_cov, model_params)
