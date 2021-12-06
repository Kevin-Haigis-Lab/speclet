import re
from pathlib import Path
from typing import Any, Optional

import pytest
import typer
from typer.testing import CliRunner

import src.command_line_interfaces.sampling_pymc3_models_cli as sampling
from src import model_configuration as model_config
from src.misc.test_helpers import assert_dicts, do_nothing
from src.model_configuration import ModelConfigurationNotFound
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

    method_calls: list[ModelFitMethod] = []

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

    advi_kwargs = {"n_iterations": 42, "draws": 23, "prior_pred_samples": 12}
    mcmc_kwargs = {
        "prior_pred_samples": 121,
        "sample_kwargs": {"draws": 29, "target_accept": 0.73},
    }

    def check_kwargs(*args: Any, **kwargs: Any) -> None:
        if fit_method is ModelFitMethod.ADVI:
            assert_dicts(advi_kwargs, kwargs)
        elif fit_method is ModelFitMethod.MCMC:
            assert_dicts(mcmc_kwargs, kwargs)
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
            pipelines={SpecletPipeline.FITTING: [ModelFitMethod.MCMC]},
            sampling_arguments={
                SpecletPipeline.FITTING.value: {
                    ModelFitMethod.ADVI.value: advi_kwargs,
                    ModelFitMethod.MCMC.value: mcmc_kwargs,
                },
                SpecletPipeline.SBC.value: {
                    ModelFitMethod.ADVI.value: {},
                    ModelFitMethod.MCMC.value: {},
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


@pytest.mark.slow
def test_print_custom_progress_messages(
    app: typer.Typer,
    runner: CliRunner,
    mock_model_config: Path,
    tmp_path: Path,
) -> None:
    res = runner.invoke(
        app,
        [
            "sampling-pymc3-models-cli_short-sampling-test",
            mock_model_config.as_posix(),
            "MCMC",
            tmp_path.as_posix(),
            "--mcmc-chains=2",
            "--mcmc-cores=1",
        ],
    )
    assert res.exit_code == 0

    n_draw_prints = 0
    for line in res.stdout.split("\n"):
        if re.search("chain \\d, draw \\d+", line):
            n_draw_prints += 1
    assert n_draw_prints == 12
