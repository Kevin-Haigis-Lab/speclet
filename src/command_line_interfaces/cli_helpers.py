#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


from pathlib import Path
from typing import Dict, Optional, Type, TypeVar, Union

import pretty_errors

from src.loggers import logger
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_five import SpecletFive, SpecletFiveConfiguration
from src.models.speclet_four import SpecletFour, SpecletFourConfiguration
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_seven import SpecletSeven, SpecletSevenConfiguration
from src.models.speclet_six import SpecletSix, SpecletSixConfiguration
from src.models.speclet_two import SpecletTwo
from src.pipelines import pipeline_classes
from src.pipelines.pipeline_classes import ModelOption

#### ---- Pretty Errors ---- ####


def configure_pretty() -> None:
    """Configure 'pretty' for better Python error messages."""
    pretty_errors.configure(
        filename_color=pretty_errors.BLUE,
        code_color=pretty_errors.BLACK,
        exception_color=pretty_errors.BRIGHT_RED,
        exception_arg_color=pretty_errors.RED,
        line_color=pretty_errors.BRIGHT_BLACK,
    )


#### ---- Models ---- ####


def clean_model_names(n: str) -> str:
    """Clean a custom model name.

    Args:
        n (str): Custom model name.

    Returns:
        str: Cleaned model name.
    """
    return n.replace(" ", "-")


_SpecletProjectModelTypes = Union[
    SpecletTestModel,
    CeresMimic,
    SpecletOne,
    SpecletTwo,
    SpecletFour,
    SpecletFive,
    SpecletSix,
    SpecletSeven,
]


def get_model_class(model_opt: ModelOption) -> Type[_SpecletProjectModelTypes]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: Dict[ModelOption, Type[_SpecletProjectModelTypes]] = {
        ModelOption.SPECLET_TEST_MODEL: SpecletTestModel,
        ModelOption.CRC_CERES_MIMIC: CeresMimic,
        ModelOption.SPECLET_ONE: SpecletOne,
        ModelOption.SPECLET_TWO: SpecletTwo,
        ModelOption.SPECLET_FOUR: SpecletFour,
        ModelOption.SPECLET_FIVE: SpecletFive,
        ModelOption.SPECLET_SIX: SpecletSix,
        ModelOption.SPECLET_SEVEN: SpecletSeven,
    }
    return model_option_map[model_opt]


#### ---- Model configurations ---- ####

_ConfigurationT = TypeVar(
    "_ConfigurationT",
    SpecletFiveConfiguration,
    SpecletFourConfiguration,
    SpecletSixConfiguration,
    SpecletSevenConfiguration,
)


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


def get_configuration_for_model(
    config_path: Path, name: str
) -> Optional[pipeline_classes.ModelConfig]:
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
        for config in pipeline_classes.get_model_configurations(
            config_path
        ).configurations
        if config.name == name
    ]
    if len(configs) > 1:
        raise FoundMultipleModelConfigurationsError(name, len(configs))
    if len(configs) == 0:
        return None
    return configs[0]


def configure_model(model: SpecletModel, name: str, config_path: Path) -> None:
    """Apply model-specific configuration from a configuration file.

    Configuration is applied in-place to the provided SpecletModel object.

    Args:
        model (SpecletModel): Speclet model to configure.
        name (str): Identifiable name of the model.
        config_path (Path): Path to the configuration file.
    """
    configuration = get_configuration_for_model(config_path, name=name)
    if configuration is not None:
        logger.info(f"Found configuration for model name: '{name}'.")
        model.set_config(configuration.config)
    else:
        logger.info(f"No configuration found for model name: '{name}'.")


def instantiate_and_configure_model(
    model_opt: ModelOption,
    name: str,
    root_cache_dir: Path,
    debug: bool,
    config_path: Path,
) -> _SpecletProjectModelTypes:
    """Instantiate and configure a model.

    Args:
        model_opt (ModelOption): Type of speclet model to create.
        name (str): Name of the model.
        root_cache_dir (Path): Root caching directory.
        debug (bool): Debug mode?
        config_path (Path): Path to configuration file.

    Returns:
        _SpecletProjectModelTypes: An instance of the desired speclet model.
    """
    ModelClass = get_model_class(model_opt=model_opt)
    speclet_model = ModelClass(name=name, root_cache_dir=root_cache_dir, debug=debug)
    configure_model(model=speclet_model, name=name, config_path=config_path)
    return speclet_model
