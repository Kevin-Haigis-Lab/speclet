from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from src import project_enums
from src.command_line_interfaces import simulation_based_calibration_cli as sbc_cli
from src.models.speclet_pipeline_test_model import SpecletTestModel


@pytest.fixture(scope="class")
def run_sbc_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(sbc_cli.run_sbc)
    return app


runner = CliRunner()


@pytest.mark.parametrize(
    "model_name", ["my-test-model", "second-test-model", "no-config-test"]
)
@pytest.mark.parametrize("fit_method", [a.value for a in project_enums.ModelFitMethod])
def test_run_sbc_with_sampling(
    model_name: str,
    fit_method: str,
    run_sbc_app: typer.Typer,
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
        run_sbc_app,
        [
            model_name,
            mock_model_config.as_posix(),
            fit_method,
            tmp_path.as_posix(),
            "111",
            "small",
        ],
    )
    assert result.exit_code == 0
