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


class ProjectConfig(BaseModel):
    """Project configurations."""

    munge: MungeConfig
    modeling: ModelingConfig


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

    with open(path, "r") as file:
        config_yaml = yaml.safe_load(file)

    config = ProjectConfig(**config_yaml)
    return config
