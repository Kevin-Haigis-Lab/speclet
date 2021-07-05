#!/usr/bin/env python3

"""CLI for checking a model configuration file."""

from pathlib import Path

import typer

from src.io import model_config as c
from src.models.configuration import instantiate_and_configure_model_from_config


def check_model_configuration(path: Path) -> None:
    """Check a model configuration file.

    Performs the following checks:

    1. Configuration file can be read in and parsed into the data structure.
    2. All names are unique.

    Args:
        path (Path): Path to the config file.
    """
    typer.secho(f"Checking model config: '{path.as_posix()}'", fg=typer.colors.BLUE)

    configs = c.get_model_configurations(path)
    typer.echo("Configuration file can be parsed: ✔︎")

    c.check_model_names_are_unique(configs)
    typer.echo("Configuration names are unique: ✔︎")

    for config in configs.configurations:
        _ = instantiate_and_configure_model_from_config(
            config,
            root_cache_dir=Path("temp"),
            debug=True,
        )
    typer.echo("All models can be instantiated and configured: ✔︎")

    typer.secho("Configuration file looks good.", fg=typer.colors.GREEN)
    return None


if __name__ == "__main__":
    typer.run(check_model_configuration)
