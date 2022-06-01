#!/usr/bin/env python3

"""Commands for general project needs."""

import logging
import os
from inspect import getdoc
from pathlib import Path
from textwrap import dedent
from typing import Optional

import typer

from speclet.bayesian_models import BayesianModel, get_bayesian_model
from speclet.loggers import set_console_handler_level

app = typer.Typer()

set_console_handler_level(logging.WARNING)


# ---- Make document with model details ----


def _write_docstring(doc: Optional[str], file: Path) -> bool:
    if doc is None:
        return False
    with open(file, "a") as f:
        f.write(dedent(doc))
        f.write("\n---\n\n")
    return True


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

    for bayesian_model in BayesianModel:
        if bayesian_model in {}:  # Add models to ignore here.
            continue

        model_cls = get_bayesian_model(bayesian_model)
        if _write_docstring(getdoc(model_cls), file=output_md):
            help()
            typer.secho(
                f"doc written for '{bayesian_model.value}'", fg=typer.colors.BLUE
            )
        else:
            typer.secho(
                f"no docstring found for '{bayesian_model.value}'", fg=typer.colors.RED
            )
    return None


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
    dirs = {
        Path("speclet"),
        Path("pipelines"),
        Path("tests"),
        Path("munge"),
        Path("data"),
    }
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
    _total = sum(line_counts.values())
    typer.secho(f"TOTAL: {_total:,}", fg=typer.colors.BRIGHT_BLUE)


if __name__ == "__main__":
    app()
