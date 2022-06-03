"""Interfacing with SLURM job scheduler."""


from datetime import timedelta as td
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from speclet.exceptions import ConfigurationNotFound
from speclet.project_enums import ModelFitMethod, SlurmPartition


class SlurmResources(BaseModel):
    """Resources for SLURM job request."""

    mem: float = 8  # GB of RAM
    time: float = 1  # hours
    cores: int = 1  # number of cores
    partition: SlurmPartition | None = None


class ResourceConfiguration(BaseModel):
    """Resource requirement configuration."""

    name: str
    active: bool = True
    slurm_resources: dict[ModelFitMethod, SlurmResources] = Field(default_factory=dict)


def read_resource_configs(config_path: Path) -> list[ResourceConfiguration]:
    """Read in SLURM resource configurations.

    Args:
        config_path (Path): Path to the configuration YAML file.

    Returns:
        list[ResourceConfiguration]: List of SLURM configurations.
    """
    with open(config_path, "r") as file:
        config_infos = yaml.safe_load(file)
    return [ResourceConfiguration(**info) for info in config_infos]


def get_resource_config(config_path: Path, name: str) -> ResourceConfiguration:
    """Get the SLURM resources for a model.

    Args:
        config_path (Path): Path to the configuration YAML file.
        name (str): Name of the model.

    Raises:
        ConfigurationNotFound: If the configuration for the model is not found.

    Returns:
        ResourceConfiguration: Resources for SLURM for the model.
    """
    with open(config_path, "r") as file:
        config_infos = yaml.safe_load(file)
    for info in config_infos:
        config = ResourceConfiguration(**info)
        if config.name == name:
            return config

    raise ConfigurationNotFound(config_path, name)


def partition_required_for_duration(time_req: td) -> SlurmPartition:
    """SLURM partition best suited for a job with a given time requirement.

    Args:
        time_req (td): Time that will be requested for the job.

    Returns:
        SlurmPartitions: The SLURM partition most appropriate for the time.
    """
    if time_req <= td(hours=12):
        return SlurmPartition.SHORT
    elif time_req <= td(days=5):
        return SlurmPartition.MEDIUM
    else:
        return SlurmPartition.LONG
