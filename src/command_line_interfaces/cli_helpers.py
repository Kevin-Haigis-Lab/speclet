#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


from pathlib import Path
from typing import Dict, Type

import pretty_errors

from src.loggers import logger
from src.models.ceres_mimic import CeresMimic
from src.models.configuration import configure_model
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_seven import SpecletSeven
from src.models.speclet_six import SpecletSix
from src.models.speclet_two import SpecletTwo
from src.project_enums import ModelOption
from src.types import SpecletProjectModelTypes

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
