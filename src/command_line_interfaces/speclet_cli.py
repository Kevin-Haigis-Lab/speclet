#!/usr/bin/env python3

"""Commands for general project needs."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pymc3 as pm
import tqdm
import typer

from src import model_configuration as model_config
from src.loggers import set_console_handler_level
from src.project_enums import MockDataSize, ModelOption

app = typer.Typer()

set_console_handler_level(logging.ERROR)


# ---- Make document with model details ----


def _write_docstring(doc: Optional[str], file: Path) -> bool:
    if doc is None:
        return False
    with open(file, "a") as f:
        f.write(doc)
        f.write("\n---\n\n")
    return True


def _remove_indents(file: Path) -> None:
    clean_lines: list[str] = []
    with open(file, "r") as open_file:
        for line in open_file:
            clean_lines.append(line.strip())
    with open(file, "w") as open_file:
        open_file.write("\n".join(clean_lines))
    return None


@app.command()
def model_docs(output_md: Optional[Path] = None, overwrite: bool = True) -> None:
    """Make a document with all SpecletModel descriptions.

    Args:
        output_md (Optional[Path], optional): Output file path. Defaults to None.
        overwrite (bool, optional): Overwrite the existing file? Defaults to True.

    Returns:
        None: None
    """
    if output_md is None:
        output_md = Path("models", "model-docs.md")

    if output_md.exists() and overwrite:
        typer.secho("Removing old file.", fg=typer.colors.BRIGHT_BLACK)
        os.remove(output_md)
    elif output_md.exists():
        typer.secho(
            "File already exists and will be added to.", fg=typer.colors.BRIGHT_BLACK
        )

    for model_opt in ModelOption:
        if model_opt in {ModelOption.SPECLET_TEST_MODEL, ModelOption.SPECLET_SIMPLE}:
            continue

        model_cls = model_config.get_model_class(model_opt)
        if _write_docstring(model_cls.__doc__, file=output_md):
            typer.secho(f"doc written for '{model_opt.value}'", fg=typer.colors.BLUE)
        else:
            typer.secho(
                f"no docstring found for '{model_opt.value}'", fg=typer.colors.RED
            )
    _remove_indents(output_md)

    return None


# ---- Make model graph images ----


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
        output_dir = Path("models", "model-graph-images")
    if config_path is None:
        config_path = model_config.get_model_config()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    configs = model_config.read_model_configurations(config_path)

    typer.echo(f"Saving images to '{output_dir.as_posix()}'.")
    typer.echo(f"Found {len(configs.configurations)} model configurations.")
    for config in tqdm.tqdm(configs.configurations):
        output_path = output_dir / config.name
        if (
            skip_existing
            and (output_path.parent / (output_path.name + ".pdf")).exists()
        ):
            continue
        sp_model = model_config.instantiate_and_configure_model(
            config, root_cache_dir=Path(tempfile.mkdtemp())
        )
        mock_data = sp_model.generate_mock_data(MockDataSize.SMALL, random_seed=1)
        sp_model.data_manager.set_data(mock_data)
        sp_model.build_model()
        mdl_graph = pm.model_to_graphviz(sp_model.model)
        mdl_graph.render(output_path.as_posix(), format="pdf", cleanup=True)
    return


# ---- Count number of lines of code ----


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
    file_types: Optional[set[str]] = None,
    ignore_dirs: Optional[set[str]] = None,
) -> int:
    n_lines = 0

    if ignore_dirs is None:
        ignore_dirs = set()

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
def lines_of_code() -> None:
    """Count the lines of code in the project."""
    # Parameters for the search
    file_types = {".py", ".smk", ".sh", ".zsh", ".R", ".r"}
    dirs = {Path("src"), Path("pipelines"), Path("tests"), Path("munge"), Path("data")}
    ignore_dirs = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}

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
