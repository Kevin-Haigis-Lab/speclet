from pathlib import Path
from typing import Any, Optional

import pytest
from typer.testing import CliRunner

from src import project_enums
from src.command_line_interfaces import simulation_based_calibration_cli as sbc_cli
from src.io import model_config
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline

runner = CliRunner()


@pytest.mark.parametrize(
    "model_name", ["my-test-model", "second-test-model", "no-config-test"]
)
@pytest.mark.parametrize("fit_method", [a.value for a in ModelFitMethod])
def test_run_sbc_with_sampling(
    model_name: str,
    fit_method: str,
    mock_model_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def do_nothing(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        SpecletTestModel, "run_simulation_based_calibration", do_nothing
    )

    result = runner.invoke(
        sbc_cli.app,
        [
            "run-sbc",
            model_name,
            mock_model_config.as_posix(),
            fit_method,
            tmp_path.as_posix(),
            "111",
            "--data-size",
            "small",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_uses_configuration_fitting_parameters(
    monkeypatch: pytest.MonkeyPatch,
    fit_method: ModelFitMethod,
    tmp_path: Path,
):

    advi_kwargs = {"n_iterations": 42, "draws": 23, "post_pred_samples": 12}
    mcmc_kwargs = {
        "tune": 33,
        "target_accept": 0.2,
        "prior_pred_samples": 121,
        "cores": 1,
    }

    def _compare_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> None:
        for k, v in d1.items():
            assert v == d2[k]

    def intercept_fit_kwargs_dict(*args, **kwargs) -> None:
        fit_kwargs: Optional[dict[Any, Any]] = kwargs["fit_kwargs"]
        assert fit_kwargs is not None
        if fit_method is ModelFitMethod.ADVI:
            _compare_dicts(advi_kwargs, fit_kwargs)
        elif fit_method is ModelFitMethod.MCMC:
            _compare_dicts(mcmc_kwargs, fit_kwargs)
        else:
            project_enums.assert_never(fit_method)

    monkeypatch.setattr(
        SpecletTestModel, "run_simulation_based_calibration", intercept_fit_kwargs_dict
    )

    model_name = "my-test-model"

    def get_mock_model_config(*args, **kwargs) -> Optional[model_config.ModelConfig]:
        return model_config.ModelConfig(
            name=model_name,
            description="",
            model=ModelOption.SPECLET_TEST_MODEL,
            fit_methods=[ModelFitMethod.ADVI],
            pipelines=[SpecletPipeline.FITTING],
            debug=False,
            pipeline_sampling_parameters={
                SpecletPipeline.SBC: {
                    ModelFitMethod.ADVI: advi_kwargs,
                    ModelFitMethod.MCMC: mcmc_kwargs,
                },
                SpecletPipeline.FITTING: {
                    ModelFitMethod.ADVI: {},
                    ModelFitMethod.MCMC: {},
                },
            },
        )

    monkeypatch.setattr(
        model_config, "get_configuration_for_model", get_mock_model_config
    )

    result = runner.invoke(
        sbc_cli.app,
        [
            "run-sbc",
            model_name,
            "not-real-config.yaml",
            fit_method.value,
            tmp_path.as_posix(),
            "111",
            "--data-size",
            "small",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
