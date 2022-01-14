from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

from speclet.command_line_interfaces.fit_bayesian_model_cli import app
from speclet.project_enums import ModelFitMethod

runner = CliRunner()


def write_config(fpath: Path) -> None:
    config_txt = """
    -   name: example-specification
        description: An example config spec.
        model: EIGHT_SCHOOLS
        data_file: DEPMAP_TEST_DATA
        sampling_kwargs:
            stan_mcmc:
                num_samples: 1000
                num_warmup: 1000
            pymc3_mcmc:
                tune: 1000
                draws: 1000
            pymc3_advi:
                n: 1000
    """
    with open(fpath, "w") as file:
        file.write(dedent(config_txt))


@pytest.mark.slow
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_app(fit_method: ModelFitMethod, tmp_path: Path) -> None:
    config_path = tmp_path / "model-config.yaml"
    write_config(config_path)
    result = runner.invoke(
        app,
        [
            "example-specification",
            str(config_path),
            fit_method.value,
            str(tmp_path),
            "--mcmc-chains=1",
            "--mcmc-cores=1",
        ],
    )
    print(result.exc_info)
    print(result.output)
    assert result.exit_code == 0
