#!/usr/bin/env python3

"""CLI for checking a model configuration file."""

import logging
from pathlib import Path

import typer

from src import model_configuration as model_config
from src.loggers import set_console_handler_level
from src.project_enums import ModelFitMethod, SpecletPipeline

set_console_handler_level(logging.WARNING)


def check_model_configuration(path: Path) -> None:
    """Check a model configuration file.

    Performs the following checks:

    1. Configuration file can be read in and parsed into the data structure.
    2. All names are unique.

    Args:
        path (Path): Path to the config file.
    """
    typer.secho(f"Checking model config: '{path.as_posix()}'", fg=typer.colors.BLUE)

    configs = model_config.read_model_configurations(path)
    typer.echo("Configuration file can be parsed: ✔︎")

    model_config.check_model_names_are_unique(configs)
    typer.echo("Configuration names are unique: ✔︎")

    for config in configs.configurations:
        _ = model_config.instantiate_and_configure_model(
            config,
            root_cache_dir=Path("temp"),
        )
    typer.echo("All models can be instantiated and configured: ✔︎")

    for config in configs.configurations:
        for fit_method in ModelFitMethod:
            for pipeline in SpecletPipeline:
                sampling_kwargs = model_config.get_sampling_kwargs_from_config(
                    config=config,
                    pipeline=pipeline,
                    fit_method=fit_method,
                )
                model_config.check_sampling_kwargs(
                    sampling_kwargs, fit_method=fit_method, pipeline=pipeline
                )
    typer.echo("All sampling parameterizations use acceptable keywords: ✔︎")

    typer.secho("Configuration file looks good.", fg=typer.colors.GREEN)
    return None


if __name__ == "__main__":
    typer.run(check_model_configuration)
