#!/usr/bin/env python3

"""Commands for general project needs."""

import logging
import tempfile
from pathlib import Path
from typing import Collection, Optional

import pymc3 as pm
import tqdm
import typer

from src.io import model_config
from src.loggers import set_console_handler_level
from src.models import configuration
from src.project_enums import MockDataSize

app = typer.Typer()

set_console_handler_level(logging.ERROR)

#### ---- Make model graph images ---- ####


@app.command()
def model_graphs(
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    skip_existing: bool = False,
) -> None:
    """Save PDFs of graphs of each model in a configuration file.

    Args:
        output_dir (Path, optional): Where to save the PDF files. Defaults to
          "models/model-graph-images".
        config_path (Optional[Path], optional): Path to a configuration file. Passing
          None (default) results in using the default configuration file for the
          project.
        skip_existing (bool, optional): Should PDFs that already exist be skipped?
          Defaults to False.
    """
    if output_dir is None:
        output_dir = Path("models/model-graph-images")
    if config_path is None:
        config_path = model_config.get_model_config()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    configs = model_config.get_model_configurations(config_path)

    typer.echo(f"Saving images to '{output_dir.as_posix()}'.")
    typer.echo(f"Found {len(configs.configurations)} model configurations.")
    for config in tqdm.tqdm(configs.configurations):
        output_path = output_dir / config.name
        if (
            skip_existing
            and (output_path.parent / (output_path.name + ".pdf")).exists()
        ):
            continue
        sp_model = configuration.instantiate_and_configure_model(
            config, root_cache_dir=Path(tempfile.mkdtemp())
        )
        mock_data = sp_model.generate_mock_data(MockDataSize.SMALL, random_seed=1)
        sp_model.data_manager.set_data(mock_data)
        sp_model.build_model()
        mdl_graph = pm.model_to_graphviz(sp_model.model)
        mdl_graph.render(output_path.as_posix(), format="pdf", cleanup=True)
    return


#### ---- Count number of lines of code ---- ####


def _count_lines_in_file(f: Path) -> int:
    n: int = 0
    try:
        with open(f) as file:
            for line in file:
                if line.strip():
                    n += 1
    except UnicodeDecodeError:
        typer.secho(
            f"Warning: unable to read file {f.as_posix()}",
            fg=typer.colors.RED,
            err=True,
        )
    return n


def _recursively_count_lines_in_dir(
    dir: Path,
    file_types: Optional[Collection[str]] = None,
    ignore_dirs: Optional[Collection[str]] = None,
) -> int:
    n_lines = 0

    if ignore_dirs is None:
        ignore_dirs = []

    for p in dir.iterdir():
        if p.name in ignore_dirs:
            continue
        if p.is_dir():
            n_lines += _recursively_count_lines_in_dir(
                p, file_types=file_types, ignore_dirs=ignore_dirs
            )
        elif p.is_file():
            if (file_types is None) or (p.suffix in file_types):
                n_lines += _count_lines_in_file(p)
    return n_lines


@app.command()
def lines_of_code(
    file_types: Collection[str] = (".py", ".smk", ".sh", ".zsh", ".R", ".r"),
    dirs: Collection[Path] = (
        Path("src"),
        Path("pipelines"),
        Path("tests"),
        Path("munge"),
        Path("data"),
    ),
    ignore_dirs: Optional[Collection[str]] = (
        "__pycache__",
        ".ipynb_checkpoints",
        ".DS_Store",
    ),
) -> None:
    """Count the lines of code.

    Args:
        file_types (list[str], optional): File types to include. Defaults to list of
          common file types for code.
        dirs (list[Path], optional): Directories to search recursively.
        ignore_dirs (list[str], optional): Names of subdirectories to ignore. Defaults
          to some standard caching and checkpoint directories.
    """
    line_counts: dict[Path, int] = {d: -1 for d in dirs}
    for dir in dirs:
        if not dir.exists() and dir.is_dir():
            typer.secho(f"Unable to analyze files in directory '{dir}'")
            continue
        line_counts[dir] = _recursively_count_lines_in_dir(
            dir, file_types=file_types, ignore_dirs=ignore_dirs
        )

    typer.secho("Lines of code:", fg=typer.colors.BRIGHT_BLUE)
    for dir, n_lines in line_counts.items():
        typer.secho(
            f"  {dir.as_posix().ljust(10)} -  {n_lines:,}", fg=typer.colors.BLUE
        )


if __name__ == "__main__":
    app()
