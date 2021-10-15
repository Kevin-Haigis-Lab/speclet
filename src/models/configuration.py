"""Handle model configuration."""

from pathlib import Path
from typing import Any

from src.io import model_config
from src.loggers import logger
from src.misc import check_kwarg_dict
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_eight import SpecletEight
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_seven import SpecletSeven
from src.models.speclet_simple import SpecletSimple
from src.models.speclet_six import SpecletSix
from src.models.speclet_two import SpecletTwo
from src.project_enums import ModelFitMethod, ModelOption, assert_never
from src.types import SpecletProjectModelTypes


def configure_model(model: SpecletModel, config_path: Path) -> None:
    """Apply model-specific configuration from a configuration file.

    Configuration is applied in-place to the provided SpecletModel object.

    Args:
        model (SpecletModel): Speclet model to configure.
        config_path (Path): Path to the configuration file.
    """
    configuration = model_config.get_configuration_for_model(
        config_path, name=model.name
    )
    if configuration is not None and configuration.config is not None:
        logger.info(f"Found configuration for model name: '{model.name}'.")
        model.set_config(configuration.config)
    else:
        logger.info(f"No configuration found for model name: '{model.name}'.")


def get_model_class(model_opt: ModelOption) -> type[SpecletProjectModelTypes]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: dict[ModelOption, type[SpecletProjectModelTypes]] = {
        ModelOption.SPECLET_TEST_MODEL: SpecletTestModel,
        ModelOption.CRC_CERES_MIMIC: CeresMimic,
        ModelOption.SPECLET_SIMPLE: SpecletSimple,
        ModelOption.SPECLET_ONE: SpecletOne,
        ModelOption.SPECLET_TWO: SpecletTwo,
        ModelOption.SPECLET_FOUR: SpecletFour,
        ModelOption.SPECLET_FIVE: SpecletFive,
        ModelOption.SPECLET_SIX: SpecletSix,
        ModelOption.SPECLET_SEVEN: SpecletSeven,
        ModelOption.SPECLET_EIGHT: SpecletEight,
    }
    return model_option_map[model_opt]


#### ---- Model configurations ---- ####


def instantiate_and_configure_model(
    config: model_config.ModelConfig, root_cache_dir: Path
) -> SpecletProjectModelTypes:
    """Create a model from a configuration.

    Args:
        config (ModelConfig): Model configuration.
        root_cache_dir (Path): Cache directory.

    Returns:
        SpecletProjectModelTypes: Configured instance of a speclet model.
    """
    logger.info("Instantiating and configuring a speclet model from config.")
    ModelClass = get_model_class(model_opt=config.model)
    speclet_model = ModelClass(
        name=config.name, root_cache_dir=root_cache_dir, debug=config.debug
    )
    if config.config is not None:
        speclet_model.set_config(config.config)
    return speclet_model


def get_config_and_instantiate_model(
    config_path: Path, name: str, root_cache_dir: Path
) -> SpecletProjectModelTypes:
    """Get a configuration and create a new instance of the speclet model.

    Args:
        config_path (Path): Path to a configuration file.
        name (str): Name of the model.
        root_cache_dir (Path): Cache directory.

    Raises:
        model_config.ModelConfigurationNotFound: Raised if no configuration file is
        found.

    Returns:
        SpecletProjectModelTypes: A configured instance of a speclet model.
    """
    config = model_config.get_configuration_for_model(config_path, name=name)
    if config is None:
        raise model_config.ModelConfigurationNotFound(name)
    speclet_model = instantiate_and_configure_model(
        config,
        root_cache_dir=root_cache_dir,
    )
    return speclet_model


def check_sampling_kwargs(
    sampling_kwargs: dict[str, Any], fit_method: ModelFitMethod
) -> None:
    """Check that sampling keyword arguments are appropriate for the method called.

    Checks a dictionary of keyword arguments against the actual parameter names in the
    expected `SpecletModel` method. If there are spare arguments, then a
    `KeywordsNotInCallableParametersError` will be raised.

    Args:
        sampling_kwargs (Dict[str, Any]): Keyword arguments to be used for sampling.
        fit_method (ModelFitMethod): Fit method to be used.
    """
    keys = list(sampling_kwargs.keys())
    blacklist = ("self",)
    if fit_method is ModelFitMethod.ADVI:
        check_kwarg_dict.check_kwarg_dict(
            keys, SpecletModel.advi_sample_model, blacklist=blacklist
        )
    elif fit_method is ModelFitMethod.MCMC:
        check_kwarg_dict.check_kwarg_dict(
            keys, SpecletModel.mcmc_sample_model, blacklist=blacklist
        )
    else:
        assert_never(fit_method)
