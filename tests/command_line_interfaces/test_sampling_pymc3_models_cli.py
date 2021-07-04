import pytest
import typer
from typer.testing import CliRunner

import src.command_line_interfaces.sampling_pymc3_models_cli as sampling
from src.project_enums import ModelOption

#### ---- CLI ---- ####


class TestTyperCLI:
    @pytest.fixture(scope="class")
    def app(self) -> typer.Typer:
        app = typer.Typer()
        app.command()(sampling.sample_speclet_model)
        return app

    @pytest.fixture(scope="class")
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_show_help(self, app: typer.Typer, runner: CliRunner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Arguments:" in result.output
        assert "Options:" in result.output

    def test_no_input_error(self, app: typer.Typer, runner: CliRunner):
        result = runner.invoke(app, [])
        assert "Error: Missing argument" in result.output
        assert result.exit_code > 0

    def test_not_real_model_error(self, app: typer.Typer, runner: CliRunner):
        result = runner.invoke(app, ["fake-model"])
        assert "Error: Invalid value" in result.output
        assert result.exit_code > 0

    def test_no_name_error(self, app: typer.Typer, runner: CliRunner):
        result = runner.invoke(app, [ModelOption.SPECLET_TEST_MODEL])
        assert "Error: Missing argument" in result.output
        assert result.exit_code > 0
