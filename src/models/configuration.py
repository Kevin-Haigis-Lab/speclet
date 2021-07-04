"""Handle model configuration."""

from pathlib import Path
from typing import Dict, Type

from src.io.model_config import get_configuration_for_model
from src.loggers import logger
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_seven import SpecletSeven
from src.models.speclet_six import SpecletSix
from src.models.speclet_two import SpecletTwo
from src.project_enums import ModelOption
from src.types import SpecletProjectModelTypes


def configure_model(model: SpecletModel, config_path: Path) -> None:
    """Apply model-specific configuration from a configuration file.

    Configuration is applied in-place to the provided SpecletModel object.

    Args:
        model (SpecletModel): Speclet model to configure.
        name (str): Identifiable name of the model.
        config_path (Path): Path to the configuration file.
    """
    configuration = get_configuration_for_model(config_path, name=model.name)
    if configuration is not None:
        logger.info(f"Found configuration for model name: '{model.name}'.")
        model.set_config(configuration.config)
    else:
        logger.info(f"No configuration found for model name: '{model.name}'.")


def get_model_class(model_opt: ModelOption) -> Type[SpecletProjectModelTypes]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: Dict[ModelOption, Type[SpecletProjectModelTypes]] = {
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


def instantiate_and_configure_model(
    model_opt: ModelOption,
    name: str,
    root_cache_dir: Path,
    debug: bool,
    config_path: Path,
) -> SpecletProjectModelTypes:
    """Instantiate and configure a model.

    Args:
        model_opt (ModelOption): Type of speclet model to create.
        name (str): Name of the model.
        root_cache_dir (Path): Root caching directory.
        debug (bool): Debug mode?
        config_path (Path): Path to configuration file.

    Returns:
        SpecletProjectModelTypes: An instance of the desired speclet model.
    """
    logger.info(f"Instantiating and configuring a '{model_opt.value}' model.")
    ModelClass = get_model_class(model_opt=model_opt)
    speclet_model = ModelClass(name=name, root_cache_dir=root_cache_dir, debug=debug)
    logger.info(f"Configuring model using config: '{config_path.as_posix()}'")
    configure_model(model=speclet_model, config_path=config_path)
    return speclet_model
