from pathlib import Path

import pytest
from typer.testing import CliRunner

from speclet.command_line_interfaces.fit_bayesian_model_cli import app
from speclet.project_enums import ModelFitMethod

runner = CliRunner()


@pytest.fixture
def config_path() -> str:
    return str(Path(__file__).parent / "fit_bayesian_model_cli_config.yaml")


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_app(fit_method: ModelFitMethod, config_path: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "example-specification",
            config_path,
            fit_method.value,
            str(tmp_path),
            "--mcmc-chains=1",
            "--mcmc-cores=1",
        ],
    )
    assert result.exit_code == 0
