"""Interfacing with SLURM job scheduler."""


from datetime import timedelta as td
from enum import Enum
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, Field, PositiveInt

from speclet.exceptions import ConfigurationNotFound
from speclet.project_enums import ModelFitMethod, SlurmPartition


class GPUModule(str, Enum):
    """Available GPU modules."""

    RTX8000 = "RTX 8000"
    TESLAV100S = "Tesla V100s"


class GPUResource(BaseModel):
    """GPU resource information."""

    gpu: GPUModule | None = None  # type of GPU
    cores: PositiveInt = 1  # number of GPUs


class SlurmResources(BaseModel):
    """Resources for SLURM job request."""

    mem: float = 8  # GB of RAM
    time: float = 1  # hours
    cores: int = 1  # number of cores
    gpu: GPUResource | None  # GPU resources
    partition: SlurmPartition | None = None  # SLURM partition


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
    with open(config_path) as file:
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
    with open(config_path) as file:
        config_infos = yaml.safe_load(file)
    for info in config_infos:
        config = ResourceConfiguration(**info)
        if config.name == name:
            return config

    raise ConfigurationNotFound(config_path, name)


def determine_necessary_partition(
    time_req: td, gpu: GPUResource | None
) -> SlurmPartition:
    """SLURM partition best suited for a job.

    Args:
        time_req (td): Time that will be requested for the job.
        gpu (GPUModule | bool | None): Requested GPU resource.

    Returns:
        SlurmPartitions: The SLURM partition most appropriate for the time.
    """
    if gpu is None or (isinstance(gpu, bool) and not gpu):
        if time_req <= td(hours=12):
            return SlurmPartition.SHORT
        elif time_req <= td(days=5):
            return SlurmPartition.MEDIUM
        else:
            return SlurmPartition.LONG
    else:
        return SlurmPartition.GPU_QUAD


_GPU_GRES_NAMES: Final[dict[GPUModule, str]] = {
    GPUModule.RTX8000: "rtx8000",
    GPUModule.TESLAV100S: "teslaV100s",
}


def get_gres_name(gpu_module: GPUModule) -> str:
    """Get the name for a GPU module recognized in the `gres` SLURM parameter.

    Args:
        gpu_module (GPUModule): GPU module.

    Returns:
        str: Name recognized in the `gres` SLURM parameter.
    """
    return _GPU_GRES_NAMES[gpu_module]
