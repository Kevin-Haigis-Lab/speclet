"""Models and enums for the pipelines."""

from collections import Counter
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml
from pydantic import BaseModel

from src.project_enums import ModelFitMethod

#### ---- Enums ---- ####


@unique
class ModelOption(str, Enum):
    """Model options."""

    SPECLET_TEST_MODEL = "speclet-test-model"
    CRC_CERES_MIMIC = "crc-ceres-mimic"
    SPECLET_ONE = "speclet-one"
    SPECLET_TWO = "speclet-two"
    SPECLET_FOUR = "speclet-four"
    SPECLET_FIVE = "speclet-five"
    SPECLET_SIX = "speclet-six"
    SPECLET_SEVEN = "speclet-seven"


@unique
class SlurmPartitions(str, Enum):
    """Partitions of the HPC available through SLURM."""

    PRIORITY = "priority"
    INTERACTIVE = "interactive"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@unique
class SpecletPipeline(Enum):
    """Pipelines available in this project."""

    FITTING = "fitting"
    SBC = "sbc"


#### ---- Models ---- ####


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    description: str
    model: ModelOption
    fit_methods: List[ModelFitMethod] = []
    config: Dict[str, Any]
    pipelines: List[SpecletPipeline]


class ModelConfigs(BaseModel):
    """Model configurations."""

    configurations: List[ModelConfig]


def get_model_configurations(path: Path) -> ModelConfigs:
    """Get a model's configuration.

    Args:
        path (Path): Path to the configuration file.

    Returns:
        ModelConfigs: Configuration spec for a model.
    """
    with open(path, "r") as file:
        return ModelConfigs(configurations=yaml.safe_load(file))


class ModelNamesAreNotAllUnique(BaseException):
    """Model names are not all unique."""

    def __init__(self, nonunique_ids: Set[str]) -> None:
        """Create a ModelNamesAreNotAllUnique error.

        Args:
            nonunique_ids (Set[str]): Set of non-unique IDs.
        """
        self.nonunique_ids = nonunique_ids
        self.message = f"Non-unique IDs: {', '.join(nonunique_ids)}"
        super().__init__(self.message)


def check_model_names_are_unique(configs: ModelConfigs) -> None:
    """Check that all model names are unique in a collection of configurations.

    Args:
        configs (ModelConfigs): Configurations to check.

    Raises:
        ModelNamesAreNotAllUnique: Raised if there are some non-unique names.
    """
    counter = Counter([config.name for config in configs.configurations])
    if not all(i == 1 for i in counter.values()):
        nonunique_ids = set([v for v, c in counter.items() if c != 1])
        raise ModelNamesAreNotAllUnique(nonunique_ids)
