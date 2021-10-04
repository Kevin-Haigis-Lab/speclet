from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import typer
from typer.testing import CliRunner

import src.command_line_interfaces.sampling_pymc3_models_cli as sampling
from src.io import model_config
from src.io.model_config import ModelConfigurationNotFound
from src.misc.test_helpers import do_nothing
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline, assert_never

#### ---- CLI ---- ####


@pytest.fixture(scope="class")
def app() -> typer.Typer:
    app = typer.Typer()
    app.command()(sampling.sample_speclet_model)
    return app


@pytest.fixture(scope="class")
def runner() -> CliRunner:
    return CliRunner()


def test_show_help(app: typer.Typer, runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Arguments:" in result.output
    assert "Options:" in result.output


def test_no_input_error(app: typer.Typer, runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    assert "Error: Missing argument" in result.output
    assert result.exit_code > 0


@pytest.mark.parametrize("model_name", ("fake-model", "not a real model", "no-model"))
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_not_real_model_name_error(
    app: typer.Typer,
    runner: CliRunner,
    model_name: str,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelConfigurationNotFound):
        _ = runner.invoke(
            app,
            [
                model_name,
                mock_model_config.as_posix(),
                fit_method.value,
                tmp_path.as_posix(),
            ],
        )


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_touch_file(
    app: typer.Typer,
    runner: CliRunner,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SpecletTestModel, "build_model", do_nothing)
    touch_path = tmp_path / "touch-file-for-testing-cli.txt"
    result = runner.invoke(
        app,
        [
            "my-test-model",
            mock_model_config.as_posix(),
            fit_method.value,
            tmp_path.as_posix(),
            "--no-sample",
            "--touch",
            touch_path.as_posix(),
        ],
    )
    assert result.exit_code == 0
    assert touch_path.exists() and touch_path.is_file()


@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize(
    "sampling,is_sampling", (("--sample", True), ("--no-sample", False))
)
def test_control_sampling(
    app: typer.Typer,
    runner: CliRunner,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sampling: str,
    is_sampling: bool,
) -> None:

    method_calls: List[ModelFitMethod] = []

    def advi_sample_model_called(*args: Any, **kwargs: Any) -> None:
        method_calls.append(ModelFitMethod.ADVI)

    def mcmc_sample_model_called(*args: Any, **kwargs: Any) -> None:
        method_calls.append(ModelFitMethod.MCMC)

    monkeypatch.setattr(SpecletTestModel, "build_model", do_nothing)
    monkeypatch.setattr(SpecletTestModel, "advi_sample_model", advi_sample_model_called)
    monkeypatch.setattr(SpecletTestModel, "mcmc_sample_model", mcmc_sample_model_called)
    result = runner.invoke(
        app,
        [
            "my-test-model",
            mock_model_config.as_posix(),
            fit_method.value,
            tmp_path.as_posix(),
            sampling,
        ],
    )
    assert result.exit_code == 0
    if not is_sampling:
        assert len(method_calls) == 0
    else:
        assert len(method_calls) == 1
        assert method_calls[0] is fit_method


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_uses_configuration_fitting_parameters(
    app: typer.Typer,
    runner: CliRunner,
    mock_model_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    fit_method: ModelFitMethod,
    tmp_path: Path,
) -> None:

    advi_kwargs = {"n_iterations": 42, "draws": 23, "post_pred_samples": 12}
    mcmc_kwargs = {"tune": 33, "target_accept": 0.2, "prior_pred_samples": 121}

    def _compare_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
        for k, v in d1.items():
            assert v == d2[k]

    def check_kwargs(*args: Any, **kwargs: Any) -> None:
        if fit_method is ModelFitMethod.ADVI:
            _compare_dicts(advi_kwargs, kwargs)
        elif fit_method is ModelFitMethod.MCMC:
            _compare_dicts(mcmc_kwargs, kwargs)
        else:
            assert_never(fit_method)

    monkeypatch.setattr(SpecletTestModel, "advi_sample_model", check_kwargs)
    monkeypatch.setattr(SpecletTestModel, "mcmc_sample_model", check_kwargs)

    model_name = "my-test-model"

    def get_mock_model_config(
        *args: Any, **kwargs: Any
    ) -> Optional[model_config.ModelConfig]:
        return model_config.ModelConfig(
            name=model_name,
            description="",
            model=ModelOption.SPECLET_TEST_MODEL,
            fit_methods=[ModelFitMethod.ADVI],
            pipelines=[SpecletPipeline.FITTING],
            debug=False,
            pipeline_sampling_parameters={
                SpecletPipeline.FITTING: {
                    ModelFitMethod.ADVI: advi_kwargs,
                    ModelFitMethod.MCMC: mcmc_kwargs,
                },
                SpecletPipeline.SBC: {
                    ModelFitMethod.ADVI: {},
                    ModelFitMethod.MCMC: {},
                },
            },
        )

    monkeypatch.setattr(
        model_config, "get_configuration_for_model", get_mock_model_config
    )

    result = runner.invoke(
        app,
        [
            model_name,
            mock_model_config.as_posix(),
            fit_method.value,
            tmp_path.as_posix(),
        ],
    )
    assert result.exit_code == 0
