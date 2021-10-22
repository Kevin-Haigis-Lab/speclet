"""Read project configuration."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from src.io.data_io import project_root_dir


class MungeConfig(BaseModel):
    """Munge configurations."""

    test: bool
    temporary_directory: Path


class ModelingConfig(BaseModel):
    """Modeling configurations."""

    highest_density_interval: float
    models_config: Path


class FittingPipelineConfig(BaseModel):
    """Fitting pipeline configurations."""

    debug: bool


class ProjectConfig(BaseModel):
    """Project configurations."""

    munge: MungeConfig
    modeling: ModelingConfig
    fitting_pipeline: FittingPipelineConfig


def read_project_configuration(path: Optional[Path] = None) -> ProjectConfig:
    """Read the project configuration.

    Args:
        path (Optional[Path], optional): Path to the configuration file if different
          from the default location. Defaults to "project-config.yaml" in the project
          root directory.

    Returns:
        ProjectConfig: Project configurations broken down by project sections.
    """
    if path is None:
        path = project_root_dir() / "project-config.yaml"

    with open(path) as file:
        config_yaml = yaml.safe_load(file)

    config = ProjectConfig(**config_yaml)
    return config


# ---- Access helpers ----


def fitting_pipeline_debug_status() -> bool:
    """Retrieve the fitting pipeline's debug status.

    Returns:
        bool: Whether or not the pipeline is in debug mode.
    """
    return read_project_configuration().fitting_pipeline.debug
