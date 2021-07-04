"""Handle model configuration."""


from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel

from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    description: str
    model: ModelOption
    fit_methods: List[ModelFitMethod] = []
    config: Dict[str, Union[ModelFitMethod, str, bool, int, float]]
    pipelines: List[SpecletPipeline]


class ModelConfigs(BaseModel):
    """Model configurations."""

    configurations: List[ModelConfig]


class FoundMultipleModelConfigurationsError(BaseException):
    """Found multiple model configurations."""

    def __init__(self, name: str, n_configs: int) -> None:
        """Create a FoundMultipleModelConfigurationsError error.

        Args:
            name (str): Name of the model whose configuration was searched for.
            n_configs (int): Number of configs found.
        """
        self.name = name
        self.n_configs = n_configs
        self.message = f"Found {self.n_configs} configuration files for '{self.name}'."
        super().__init__(self.message)


def get_model_configurations(path: Path) -> ModelConfigs:
    """Get a model's configuration.

    Args:
        path (Path): Path to the configuration file.

    Returns:
        ModelConfigs: Configuration spec for a model.
    """
    with open(path, "r") as file:
        return ModelConfigs(configurations=yaml.safe_load(file))


def get_configuration_for_model(config_path: Path, name: str) -> Optional[ModelConfig]:
    """Get the configuration information for a named model.

    Args:
        config_path (Path): Path to the configuration file.
        name (str): Model configuration identifier.

    Raises:
        FoundMultipleModelConfigurationsError: Raised if multiple configuration files
        are found.

    Returns:
        Optional[pipeline_classes.ModelConfig]: If a configuration file is found, it is
        returned, else None.
    """
    configs = [
        config
        for config in get_model_configurations(config_path).configurations
        if config.name == name
    ]
    if len(configs) > 1:
        raise FoundMultipleModelConfigurationsError(name, len(configs))
    if len(configs) == 0:
        return None
    return configs[0]


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
