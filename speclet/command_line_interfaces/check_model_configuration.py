#!/usr/bin/env python3

"""CLI for checking a model configuration file."""

import logging
from pathlib import Path
from typing import Optional

from typer import Typer, colors, secho

from speclet import model_configuration as model_config
from speclet.bayesian_models import get_bayesian_model
from speclet.loggers import set_console_handler_level
from speclet.project_configuration import get_model_configuration_file

set_console_handler_level(logging.WARNING)

app = Typer()


@app.command()
def check_model_configuration(path: Optional[Path] = None) -> None:
    """Check a model configuration file.

    Performs the following checks:

    1. Configuration file can be read in and parsed into the data structure.
    2. All names are unique.

    Args:
        path (Optional[Path], optional): Path to the config file. Defaults to None
    """
    if path is None:
        path = get_model_configuration_file()
    secho(f"Checking model config: '{str(path)}'", fg=colors.BLUE)

    secho("Trying to parse configuration file...")
    configs = model_config.read_model_configurations(path)
    secho("Configuration file can be parsed: ✔︎", fg=colors.GREEN)

    secho("Checking all names are unique...")
    model_config.check_model_names_are_unique(configs)
    secho("Configuration names are unique: ✔︎", fg=colors.GREEN)

    secho("Checking all models can be instantiated and configured...")
    for config in configs.configurations:
        secho(f"  {config.name}", fg=colors.BRIGHT_BLACK)
        _ = get_bayesian_model(config.model)()

    secho("All models can be instantiated and configured: ✔︎", fg=colors.GREEN)

    secho("Configuration file looks good.", fg=colors.BLUE, bold=True)
    return None


if __name__ == "__main__":
    app()
