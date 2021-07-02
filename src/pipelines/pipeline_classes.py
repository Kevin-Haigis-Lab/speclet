"""Models and enums for the pipelines."""


from enum import Enum, unique
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

from src.project_enums import ModelFitMethod

#### ---- Enums ---- ####


@unique
class ModelOption(str, Enum):
    """Model options."""

    speclet_test_model = "speclet-test-model"
    crc_ceres_mimic = "crc-ceres-mimic"
    speclet_one = "speclet-one"
    speclet_two = "speclet-two"
    speclet_four = "speclet-four"
    speclet_five = "speclet-five"
    speclet_six = "speclet-six"
    speclet_seven = "speclet-seven"


@unique
class SlurmPartitions(str, Enum):
    """Partitions of the HPC available through SLURM."""

    priority = "priority"
    interactive = "interactive"
    short = "short"
    medium = "medium"
    long = "long"


#### ---- Models ---- ####


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    model: ModelOption
    fit_method: ModelFitMethod


class ModelConfigs(BaseModel):
    """Model configurations."""

    configurations: List[ModelConfig]


def model_config_from_yaml(path: Path) -> ModelConfigs:
    """Read in model configurations.

    Args:
        path (Path): YAML configuration file.

    Returns:
        ModelConfig: List of model configurations.
    """
    with open(path, "r") as file:
        config = ModelConfigs(**yaml.safe_load(file))
    return config
