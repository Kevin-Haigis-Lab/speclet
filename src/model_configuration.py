"""Handle model configuration."""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import pymc3 as pm
import yaml
from pydantic import BaseModel, Field, PositiveInt
from pydantic.types import confloat

from src.io.data_io import project_root_dir
from src.loggers import logger
from src.modeling.pymc3_sampling_api import VIMethod
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
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline, assert_never
from src.types import BasicTypes

# ---- Types ----

SpecletProjectModelTypes = Union[
    SpecletTestModel,
    SpecletSimple,
    CeresMimic,
    SpecletOne,
    SpecletTwo,
    SpecletFour,
    SpecletFive,
    SpecletSix,
    SpecletSeven,
    SpecletEight,
]

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)

# ----  Configuration classes ----


class Pymc3SampleArguments(BaseModel):
    """Model `sample()` keyword arguments (PyMC3 v3.11.2)."""

    _pymc3_version: str = "3.11.2"
    draws: int = 1000
    step: Optional[Union[Callable, Iterable[Callable]]] = None
    init: str = "auto"
    n_init: int = 200000
    chain_idx: PositiveInt = 0
    chains: Optional[PositiveInt] = None
    cores: Optional[PositiveInt] = None
    tune: PositiveInt = 1000
    progressbar: bool = True
    random_seed: Optional[PositiveInt] = None
    discard_tuned_samples: bool = True
    compute_convergence_checks: bool = True
    return_inferencedata: Optional[bool] = True  # not default
    idata_kwargs: Optional[dict[str, BasicTypes]] = None
    target_accept: TargetAcceptFloat = 0.8  # type: ignore

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3SampleArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        if self._pymc3_version != pm.__version__:
            raise NotImplementedError(
                f"Support for {self._pymc3_version} -> current ver {pm.__version__}"
            )


class Pymc3FitArguments(BaseModel):
    """Model `fit()` keyword arguments (PyMC3 v3.11.2)."""

    _pymc3_version: str = "3.11.2"
    n: PositiveInt = 10000
    method: VIMethod = "advi"
    random_seed: Optional[PositiveInt] = None
    inf_kwargs: Optional[dict[str, BasicTypes]] = None
    progressbar: bool = True

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3FitArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        if self._pymc3_version != pm.__version__:
            raise NotImplementedError(
                f"Support for {self._pymc3_version} -> current ver {pm.__version__}"
            )


class SpecletModelMcmcArguments(BaseModel):
    """SpecletModel MCMC arguments."""

    prior_pred_samples: Optional[int] = 500
    random_seed: Optional[int] = None
    sample_kwargs: Optional[Pymc3SampleArguments] = None
    ignore_cache: bool = False


class SpecletModelAdviArguments(BaseModel):
    """SpecletModel ADVI arguments."""

    method: VIMethod = "advi"
    n_iterations: int = 100000
    draws: int = 1000
    prior_pred_samples: Optional[int] = None
    random_seed: Optional[int] = None
    fit_kwargs: Optional[dict[str, BasicTypes]] = None
    ignore_cache: bool = False


class SpecletModelSamplingArguments(BaseModel):
    """Arguments for SpecletModel fitting methods."""

    MCMC: Optional[SpecletModelMcmcArguments] = None
    ADVI: Optional[SpecletModelAdviArguments] = None


class PipelineSamplingArguments(BaseModel):
    """Sampling arguments for each pipeline."""

    fitting: Optional[SpecletModelSamplingArguments] = None
    sbc: Optional[SpecletModelSamplingArguments] = None


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    description: str
    model: ModelOption
    config: Optional[dict[str, Union[ModelFitMethod, BasicTypes]]] = None
    pipelines: dict[SpecletPipeline, list[ModelFitMethod]] = Field(default_factory=dict)
    sampling_arguments: PipelineSamplingArguments = Field(
        default_factory=PipelineSamplingArguments
    )


class ModelConfigs(BaseModel):
    """Model configurations."""

    configurations: list[ModelConfig]


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


def get_model_config() -> Path:
    """Path to default model configuration file.

    Returns:
        Path: Path to a model configuration.
    """
    return project_root_dir() / "models" / "model-configs.yaml"


def read_model_configurations(path: Path) -> ModelConfigs:
    """Get a model's configuration.

    Args:
        path (Path): Path to the configuration file.

    Returns:
        ModelConfigs: Configuration spec for a model.
    """
    with open(path) as file:
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
        for config in read_model_configurations(config_path).configurations
        if config.name == name
    ]
    if len(configs) > 1:
        raise FoundMultipleModelConfigurationsError(name, len(configs))
    if len(configs) == 0:
        return None
    return configs[0]


# ---- Model configuration ----


def configure_model(model: SpecletModel, config_path: Path) -> None:
    """Apply model-specific configuration from a configuration file.

    Configuration is applied in-place to the provided SpecletModel object.

    Args:
        model (SpecletModel): Speclet model to configure.
        config_path (Path): Path to the configuration file.
    """
    configuration = get_configuration_for_model(config_path, name=model.name)
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
    if model_opt not in model_option_map:
        raise ModelOptionNotAssociatedWithAClassException(model_opt.value)
    return model_option_map[model_opt]


def instantiate_and_configure_model(
    config: ModelConfig, root_cache_dir: Path
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
    speclet_model = ModelClass(name=config.name, root_cache_dir=root_cache_dir)
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
        ModelConfigurationNotFound: Raised if no configuration file is
        found.

    Returns:
        SpecletProjectModelTypes: A configured instance of a speclet model.
    """
    config = get_configuration_for_model(config_path, name=name)
    if config is None:
        raise ModelConfigurationNotFound(name)
    speclet_model = instantiate_and_configure_model(
        config,
        root_cache_dir=root_cache_dir,
    )
    return speclet_model


# ---- Keyword arguments from config ----


def _get_fit_method_arguments(
    pipeline_args: Optional[SpecletModelSamplingArguments], fit_method: ModelFitMethod
) -> Optional[Union[SpecletModelMcmcArguments, SpecletModelAdviArguments]]:
    if pipeline_args is None:
        return None
    elif fit_method is ModelFitMethod.MCMC:
        return pipeline_args.MCMC
    elif fit_method is ModelFitMethod.ADVI:
        return pipeline_args.ADVI
    else:
        assert_never(fit_method)


def get_sampling_kwargs_from_config(
    config: ModelConfig,
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
) -> Optional[Union[SpecletModelMcmcArguments, SpecletModelAdviArguments]]:
    """Get the sampling keyword argument dictionary from a model configuration.

    Args:
        config (ModelConfig): Model configuration to extract the dictionary from.
        pipeline (SpecletPipeline): Pipeline that is/will be used.
        fit_method (ModelFitMethod): Desired model fitting method.

    Returns:
        Optional[Union[SpecletModelMcmcArguments, SpecletModelAdviArguments]]: Model
        fitting arguments.
    """
    sampling_arguments = config.sampling_arguments
    if pipeline is SpecletPipeline.FITTING:
        return _get_fit_method_arguments(sampling_arguments.fitting, fit_method)
    elif pipeline is SpecletPipeline.SBC:
        return _get_fit_method_arguments(sampling_arguments.sbc, fit_method)
    else:
        assert_never(pipeline)


def get_sampling_kwargs(
    config_path: Path, name: str, pipeline: SpecletPipeline, fit_method: ModelFitMethod
) -> Optional[Union[SpecletModelMcmcArguments, SpecletModelAdviArguments]]:
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
        dict[str, ModelFitConfiguration]: Keyword arguments for the model-fitting
        method.
    """
    if (config := get_configuration_for_model(config_path, name)) is None:
        raise ModelConfigurationNotFound(name)
    return get_sampling_kwargs_from_config(
        config, pipeline=pipeline, fit_method=fit_method
    )


# ---- Checks ----


def check_model_names_are_unique(configs: ModelConfigs) -> bool:
    """Check that all model names are unique in a collection of configurations.

    Args:
        configs (ModelConfigs): Configurations to check.

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
