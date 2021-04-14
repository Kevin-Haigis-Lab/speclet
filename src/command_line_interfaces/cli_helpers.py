#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


from enum import Enum
from logging import Logger
from typing import Dict, Optional, Type

import pretty_errors

from src.models.crc_ceres_mimic import CrcCeresMimic
from src.models.crc_model_one import CrcModelOne
from src.models.speclet_model import SpecletModel

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


class ModelOption(str, Enum):
    """Model options."""

    crc_model_one = "crc_model_one"
    crc_ceres_mimic = "crc_ceres_mimic"


def get_model_class(model_opt: ModelOption) -> Type[SpecletModel]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: Dict[ModelOption, Type[SpecletModel]] = {
        ModelOption.crc_model_one: CrcModelOne,
        ModelOption.crc_ceres_mimic: CrcCeresMimic,
    }
    return model_option_map[model_opt]


#### ---- Modifying models ---- ####


def modify_ceres_model_by_name(
    model: CrcCeresMimic, name: str, logger: Optional[Logger] = None
) -> None:
    """Modify a CrcCeresMimic object based on the user-provided input name.

    Args:
        model (CrcCeresMimic): The CrcCeresMimic model.
        name (str): User-provided name.
        logger (Optional[Logger], optional): A logger object. Defaults to None.
    """
    if "copynumber" in name:
        if logger is not None:
            logger.info("Including gene copy number covariate in CERES model.")
        model.copynumber_cov = True
    if "sgrnaint" in name:
        if logger is not None:
            logger.info("Including sgRNA|gene varying intercept in CERES model.")
        model.sgrna_intercept_cov = True
