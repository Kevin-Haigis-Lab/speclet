"""Read project configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel

from speclet.loggers import logger


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


def read_project_configuration(path: Path | None = None) -> ProjectConfig:
    """Read the project configuration.

    Searches the .env for `PROJECT_CONFIG`.

    Args:
        path (Optional[Path], optional): Path to the configuration file if different
        from the default location. If no path is passed, searches .env for
        `PROJECT_CONFIG`.

    Returns:
        ProjectConfig: Project configurations broken down by project sections.
    """
    DEFAULT_CONFIG = Path("project-config.yaml")
    ENV_VAR = "PROJECT_CONFIG"
    if path is None:
        if (env_config := dotenv_values().get(ENV_VAR)) is not None:
            path = Path(env_config)
        elif (env_config := os.getenv(ENV_VAR)) is not None:
            path = Path(env_config)
        elif DEFAULT_CONFIG.exists():
            logger.info("Using default project configuration file.")
            path = DEFAULT_CONFIG
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


def on_hms_cluster() -> bool:
    """Determine if on the HMS cluster or not.

    Returns:
        bool: Whether the current program is running on the HMS cluster.
    """
    env_var = "HMS_CLUSTER"
    return os.getenv(env_var) is not None


def arviz_config() -> None:
    """Set common ArviZ defaults."""
    az.rcParams["stats.hdi_prob"] = get_bayesian_modeling_constants().hdi_prob
