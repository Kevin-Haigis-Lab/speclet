#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


from enum import Enum
from typing import Any, Dict, Type, Union

import pretty_errors

from src.loggers import logger
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_three import SpecletThree
from src.models.speclet_two import SpecletTwo

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

    speclet_test_model = "speclet-test-model"
    crc_ceres_mimic = "crc-ceres-mimic"
    speclet_one = "speclet-one"
    speclet_two = "speclet-two"
    speclet_three = "speclet-three"
    speclet_four = "speclet-four"
    speclet_five = "speclet-five"
    speclet_six = "speclet-six"
    speclet_seven = "speclet-seven"


def get_model_class(model_opt: ModelOption) -> Type[SpecletModel]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: Dict[ModelOption, Type[SpecletModel]] = {
        ModelOption.speclet_test_model: SpecletTestModel,
        ModelOption.crc_ceres_mimic: CeresMimic,
        ModelOption.speclet_one: SpecletOne,
        ModelOption.speclet_two: SpecletTwo,
        ModelOption.speclet_three: SpecletThree,
        ModelOption.speclet_four: SpecletFour,
        ModelOption.speclet_five: SpecletFive,
    }
    return model_option_map[model_opt]


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"


#### ---- Modifying models ---- ####


def modify_model_by_name(model: Any, name: str) -> None:
    """Modify a model using keys in the name.

    Args:
        model (Any): Any model. If it is a type that has a modification method, then it
          will be sent through the method.
        name (str): Name of the model provided by the user.

    Returns:
        [None]: None
    """
    if isinstance(model, CeresMimic):
        modify_ceres_model_by_name(model, name)
    elif isinstance(model, SpecletTwo) or isinstance(model, SpecletThree):
        modify_speclettwo_and_three_model_by_name(model, name)
    elif isinstance(model, SpecletFour):
        modify_specletfour_model_by_name(model, name)
    return None


def modify_ceres_model_by_name(model: CeresMimic, name: str) -> None:
    """Modify a CeresMimic object based on the user-provided input name.

    Args:
        model (CeresMimic): The CeresMimic model.
        name (str): User-provided name.
    """
    if "copynumber" in name:
        logger.info("Including gene copy number covariate in CERES model.")
        model.copynumber_cov = True
    if "sgrnaint" in name:
        logger.info("Including sgRNA|gene varying intercept in CERES model.")
        model.sgrna_intercept_cov = True


def modify_speclettwo_and_three_model_by_name(
    model: Union[SpecletTwo, SpecletThree], name: str
) -> None:
    """Modify a SpecletTwo/Three object based on the user-provided input name.

    Args:
        model (Union[SpecletTwo, SpecletThree]): The SpecletTwo or SpecletThree model.
        name (str): User-provided name.
    """
    if "kras" in name:
        logger.info("Including KRAS allele covariate in the model.")
        model.kras_cov = True


def modify_specletfour_model_by_name(model: SpecletFour, name: str) -> None:
    """Modify a SpecletFour object based on the user-provided input name.

    Args:
        model (SpecletFour): The SpecletFour model.
        name (str): User-provided name.
    """
    if "copy-number" in name or "cn-cov" in name:
        logger.info("Including copy number covariate in the Sp4 model.")
        model.copy_number_cov = True
