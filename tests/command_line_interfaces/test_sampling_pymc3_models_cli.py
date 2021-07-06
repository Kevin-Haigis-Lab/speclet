from pathlib import Path
from typing import Any, List

import pytest
import typer
from typer.testing import CliRunner

import src.command_line_interfaces.sampling_pymc3_models_cli as sampling
from src.io.model_config import ModelConfigurationNotFound
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.project_enums import ModelFitMethod

#### ---- CLI ---- ####


@pytest.fixture(scope="class")
def app() -> typer.Typer:
    app = typer.Typer()
    app.command()(sampling.sample_speclet_model)
    return app


@pytest.fixture(scope="class")
def runner() -> CliRunner:
    return CliRunner()


def test_show_help(app: typer.Typer, runner: CliRunner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Arguments:" in result.output
    assert "Options:" in result.output


def test_no_input_error(app: typer.Typer, runner: CliRunner):
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
):
    with pytest.raises(ModelConfigurationNotFound):
        _ = runner.invoke(
            app, [model_name, mock_model_config.as_posix(), fit_method.value, "temp"]
        )


def do_nothing(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_touch_file(
    app: typer.Typer,
    runner: CliRunner,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(SpecletTestModel, "build_model", do_nothing)
    touch_path = tmp_path / "touch-file-for-testing-cli.txt"
    result = runner.invoke(
        app,
        [
            "my-test-model",
            mock_model_config.as_posix(),
            fit_method.value,
            "temp",
            "--no-sample",
            "--touch",
            touch_path.as_posix(),
        ],
    )
    assert result.exit_code == 0
    assert touch_path.exists() and touch_path.is_file()


@pytest.mark.DEV
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
    sampling: str,
    is_sampling: bool,
):

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
            "temp",
            sampling,
        ],
    )
    assert result.exit_code == 0
    if not is_sampling:
        assert len(method_calls) == 0
    else:
        assert len(method_calls) == 1
        assert method_calls[0] is fit_method
