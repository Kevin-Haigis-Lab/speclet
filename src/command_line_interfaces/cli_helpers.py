#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


from typing import Any, Dict, Type

import pretty_errors

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
from src.pipelines.pipeline_classes import ModelOption
from src.project_enums import ModelFitMethod

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
        ModelOption.speclet_four: SpecletFour,
        ModelOption.speclet_five: SpecletFive,
        ModelOption.speclet_six: SpecletSix,
        ModelOption.speclet_seven: SpecletSeven,
    }
    return model_option_map[model_opt]


def extract_fit_method(name: str) -> ModelFitMethod:
    """Extract the model fitting method to use based on the unique name of the model.

    Args:
        name (str): Name of the model (*not* the type of model).

    Raises:
        ValueError: Raised if no fitting method is found.

    Returns:
        ModelFitMethod: The method to use for the fitting the model.
    """
    for method_name, method_member in ModelFitMethod.__members__.items():
        if method_name.lower() in name.lower():
            return method_member
    raise ValueError(f"Did not find a viable fit method in the model name: '{name}'")


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
    elif isinstance(model, SpecletFour):
        modify_specletfour_model_by_name(model, name)
    elif isinstance(model, SpecletSix):
        modify_specletsix_model_by_name(model, name)
    elif isinstance(model, SpecletSeven):
        modify_specletseven_model_by_name(model, name)
    else:
        logger.info("No modifications make to model based on its name.")
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


def modify_specletfour_model_by_name(model: SpecletFour, name: str) -> None:
    """Modify a SpecletFour object based on the user-provided input name.

    Args:
        model (SpecletFour): The SpecletFour model.
        name (str): User-provided name.
    """
    if "copy-number" in name or "cn-cov" in name:
        logger.info("Including copy number covariate in the Sp4 model.")
        model.copy_number_cov = True


def modify_specletsix_model_by_name(model: SpecletSix, name: str) -> None:
    """Modify a SpecletSix object based on the user-provided input name.

    Args:
        model (SpecletSix): The SpecletSix model.
        name (str): User-provided name.
    """
    if "cellcna" in name:
        logger.info("Including cell line copy number covariate in the Sp6 model.")
        model.cell_line_cna_cov = True
    if "genecna" in name:
        logger.info("Including gene copy number covariate in the Sp6 model.")
        model.gene_cna_cov = True
    if "rna" in name:
        logger.info("Including RNA covariate in the Sp6 model.")
        model.rna_cov = True
    if "mutation" in name:
        logger.info("Including mutation covariate in the Sp6 model.")
        model.mutation_cov = True


def modify_specletseven_model_by_name(model: SpecletSeven, name: str) -> None:
    """Modify a SpecletSeven object based on the user-provided input name.

    Args:
        model (SpecletSeven): The SpecletSeven model.
        name (str): User-provided name.
    """
    logger.info(f"Modifying SpecletSeven model based on the name: '{name}'.")
