"""Handle model configuration."""


from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel

from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline

PipelineSamplingParameters = Dict[
    SpecletPipeline, Dict[ModelFitMethod, Dict[str, Union[float, str, int]]]
]


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    description: str
    model: ModelOption
    config: Optional[Dict[str, Union[ModelFitMethod, str, bool, int, float]]]
    fit_methods: List[ModelFitMethod]
    pipelines: List[SpecletPipeline]
    debug: bool
    pipeline_sampling_parameters: Optional[PipelineSamplingParameters]


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


def get_sampling_kwargs_from_config(
    config: ModelConfig,
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
) -> Dict[str, Union[float, str, int]]:
    """Get the sampling keyword argument dictionary from a model configuration.

    Args:
        config (ModelConfig): Model configuration to extract the dictionary from.
        pipeline (SpecletPipeline): Pipeline that is/will be used.
        fit_method (ModelFitMethod): Desired model fitting method.

    Returns:
        Dict[str, Union[float, str, int]]: Keyword arguments for the model-fitting
        method.
    """
    if (sampling_params := config.pipeline_sampling_parameters) is None:
        return {}
    if (pipeline_dict := sampling_params.get(pipeline, None)) is None:
        return {}
    if (kwargs_dict := pipeline_dict.get(fit_method, None)) is None:
        return {}
    return kwargs_dict


def get_sampling_kwargs(
    config_path: Path, name: str, pipeline: SpecletPipeline, fit_method: ModelFitMethod
) -> Dict[str, Union[float, str, int]]:
    """Get the sampling keyword argument dictionary from a configuration file.

    Args:
        config_path (Path): Path to the configuration file.
        name (str): Identifiable name of the model.
        pipeline (SpecletPipeline): Pipeline that is/will be used.
        fit_method (ModelFitMethod): Desired model fitting method.

    Raises:
        ModelConfigurationNotFound: Raised if the configuration for the model name is
        not found.

    Returns:
        Dict[str, Union[float, str, int]]: Keyword arguments for the model-fitting
        method.
    """
    if (config := get_configuration_for_model(config_path, name)) is None:
        raise ModelConfigurationNotFound(name)
    return get_sampling_kwargs_from_config(
        config, pipeline=pipeline, fit_method=fit_method
    )


class ModelConfigurationNotFound(BaseException):
    """Model configuration not found."""

    def __init__(self, model_name: str) -> None:
        """Create a ModelConfigurationNotFound error.

        Args:
            model_name (str): Name of the model.
        """
        self.model_name = model_name
        self.message = f"Configuration not found for model: '{self.model_name}'."
        super().__init__(self.message)


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
