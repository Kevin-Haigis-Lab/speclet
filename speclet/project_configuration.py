"""Read project configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel


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
    num_chains: int
    models_config: Path
    temp_dir: Path
    model_cache_dir: Path
    reports_dir: Path
    benchmark_dir: Path
    env_yaml: Path


class ProjectConfig(BaseModel):
    """Project configurations."""

    munge: MungeConfig
    modeling: ModelingConfig
    fitting_pipeline: FittingPipelineConfig


def read_project_configuration(path: Optional[Path] = None) -> ProjectConfig:
    """Read the project configuration.

    Searches the .env for `PROJECT_CONFIG`.

    Args:
        path (Optional[Path], optional): Path to the configuration file if different
        from the default location. If no path is passed, searches .env for
        `PROJECT_CONFIG`.

    Returns:
        ProjectConfig: Project configurations broken down by project sections.
    """
    if path is None:
        if (env_config := dotenv_values().get("PROJECT_CONFIG")) is not None:
            path = Path(env_config)
        else:
            raise BaseException("Project configuration YAML file not found.")

    with open(path) as file:
        config_yaml = yaml.safe_load(file)

    config = ProjectConfig(**config_yaml)
    return config


# ---- Access helpers ----


def fitting_pipeline_config() -> FittingPipelineConfig:
    """The configuration for the model fitting pipeline."""
    return read_project_configuration().fitting_pipeline


@dataclass(frozen=True)
class BayesianModelingConstants:
    """PyMC3 global constants."""

    hdi_prob: float


def get_bayesian_modeling_constants() -> BayesianModelingConstants:
    """Get the global constants for use with Bayesian data analysis and modeling.

    Returns:
        BayesianAnalysisConstants: Bayesian modeling global constants.
    """
    project_config = read_project_configuration()
    return BayesianModelingConstants(
        hdi_prob=project_config.modeling.highest_density_interval
    )


def get_model_configuration_file() -> Path:
    """Get the default model configuration file for the project."""
    return read_project_configuration().modeling.models_config


# ---- Not in the YAML file ----


def on_o2() -> bool:
    """Determine if on O2 or not.

    Returns:
        bool: Whether the current program is running on O2.
    """
    hostname = os.getenv("HOSTNAME")
    if hostname is None:
        return False
    return "o2.rc.hms.harvard.edu" in hostname.lower()
