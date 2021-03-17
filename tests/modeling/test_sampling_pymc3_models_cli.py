#!/usr/bin/env python3

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from src.modeling import sampling_pymc3_models_cli as sampling_cli


class TestTyperCLI:
    @pytest.fixture(scope="class")
    def app(self) -> typer.Typer:
        app = typer.Typer()
        app.command()(sampling_cli.main)
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
        result = runner.invoke(app, [sampling_cli.ModelOption.crc_m1])
        assert "Error: Missing argument" in result.output
        assert result.exit_code > 0
