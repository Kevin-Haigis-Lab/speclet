#!/usr/bin/env python3

"""Commands for general project needs."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pymc3 as pm
import tqdm
import typer

from src.io import model_config
from src.loggers import set_console_handler_level
from src.models import configuration
from src.project_enums import MockDataSize

app = typer.Typer()

set_console_handler_level(logging.ERROR)


@app.command()
def model_graphs(
    output_dir: Path = Path("models/model-graph-images"),
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


if __name__ == "__main__":
    app()
