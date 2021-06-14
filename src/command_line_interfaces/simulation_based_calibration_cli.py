#!/usr/bin/env python3

"""A helpful command line interface for simulation-based calibration."""

from pathlib import Path

import typer

from src.command_line_interfaces import cli_helpers
from src.command_line_interfaces.cli_helpers import ModelOption
from src.loggers import logger

app = typer.Typer()
cli_helpers.configure_pretty()


@app.command()
def run_sbc(
    model_class: ModelOption,
    name: str,
    cache_dir: Path,
    sim_number: int,
    data_size: str,
) -> None:
    """CLI for running a round of simulation-based calibration for a model.

    The `name` argument can contain model specification information. These are
    documented below.

    [model: CERES Mimic]
    - If the name contains "copynumber" then the copynumber covariate will be included.
    - If the name contains "sgrnaint" then the sgRNA|gene varying intercept will be
      included.

    Args:
        model_class (ModelOption): Name of the model to use.
        name (str): Unique identifiable name for the model.
        cache_dir (Path): Where to store the results.
        sim_number (int): Simulation number.
        data_size (str): Which data size to use. See the actual methods
          for details and options.

    Returns:
        None: None
    """
    logger.info(f"Running SBC for model '{model_class.value}' named '{name}'.")
    name = cli_helpers.clean_model_names(name)
    ModelClass = cli_helpers.get_model_class(model_opt=model_class)
    model = ModelClass(
        f"{name}-sbc{sim_number}",
        root_cache_dir=cache_dir,
        debug=True,
    )
    cli_helpers.modify_model_by_name(model=model, name=name)
    model.run_simulation_based_calibration(
        cache_dir,
        fit_method=cli_helpers.extract_fit_method(name),
        random_seed=sim_number,
        size=data_size,
    )
    logger.info("SBC finished.")
    return None


if __name__ == "__main__":
    app()
