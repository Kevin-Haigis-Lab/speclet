"""Handle model configuration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from speclet import io
from speclet.bayesian_models import BayesianModel
from speclet.modeling.fitting_arguments import ModelingSamplingArguments
from speclet.project_enums import ModelFitMethod

# ----  Configuration classes ----


class PipelineChoices(BaseModel):
    """Selection of fit methods to use for a model per pipeline."""

    fitting: list[ModelFitMethod] = Field(default_factory=list)
    sbc: list[ModelFitMethod] = Field(default_factory=list)


class BayesianModelConfiguration(BaseModel):
    """Bayesian model configuration."""

    name: str
    description: str
    active: bool = True
    model: BayesianModel
    data_file: io.DataFile | Path
    sampling_kwargs: ModelingSamplingArguments = Field(
        default_factory=ModelingSamplingArguments
    )


class BayesianModelConfigurations(BaseModel):
    """Bayesian model configurations."""

    configurations: list[BayesianModelConfiguration]

    def active_only(self) -> None:
        """Filter only active configurations."""
        self.configurations = [c for c in self.configurations if c.active]
        return None


# ---- Exceptions ----


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

    def __init__(self, nonunique_ids: set[str]) -> None:
        """Create a ModelNamesAreNotAllUnique error.

        Args:
            nonunique_ids (Set[str]): Set of non-unique IDs.
        """
        self.nonunique_ids = nonunique_ids
        self.message = f"Non-unique IDs: {', '.join(nonunique_ids)}"
        super().__init__(self.message)


class ModelOptionNotAssociatedWithAClassException(BaseException):
    """Model option not yet associated with a class."""

    pass


# ---- Files and I/O ----


def read_model_configurations(
    path: Path, active_only: bool = False
) -> BayesianModelConfigurations:
    """Read the file of Bayesian model configurations.

    Args:
        path (Path): Path to the configuration file.
        active_only (bool, optional): Filter for only "active" model configurations.
        Defaults to `False`.

    Returns:
        BayesianModelConfigurations: Bayesian model configurations.
    """
    with open(path) as file:
        configs = BayesianModelConfigurations(configurations=yaml.safe_load(file))
    if active_only:
        configs.active_only()
    return configs


def get_configuration_for_model(
    config_path: Path, name: str
) -> BayesianModelConfiguration:
    """Get the configuration information for a named model.

    Args:
        config_path (Path): Path to the configuration file.
        name (str): Model configuration identifier.

    Raises:
        FoundMultipleModelConfigurationsError: If multiple configurations are found.
        ModelConfigurationNotFound: If no configuration is found.

    Returns:
        Optional[BayesianModelConfiguration]: The model configuration.
    """
    configs = read_model_configurations(config_path).configurations
    configs = [config for config in configs if config.name == name]
    if len(configs) > 1:
        raise FoundMultipleModelConfigurationsError(name, len(configs))
    if len(configs) == 0:
        raise ModelConfigurationNotFound(name)
    return configs[0]


# ---- Checks ----


def check_model_names_are_unique(configs: BayesianModelConfigurations) -> bool:
    """Check that all model names are unique in a collection of configurations.

    Args:
        configs (BayesianModelConfigurations): Configurations to check.

    Raises:
        ModelNamesAreNotAllUnique: Raised if there are some non-unique names.

    Returns:
        bool: True is model names are unique.
    """
    counter = Counter([config.name for config in configs.configurations])
    if not all(i == 1 for i in counter.values()):
        nonunique_ids = {v for v, c in counter.items() if c != 1}  # noqa: C403
        raise ModelNamesAreNotAllUnique(nonunique_ids)
    return True
