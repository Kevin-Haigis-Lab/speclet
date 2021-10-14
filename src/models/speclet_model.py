"""Base class to contain a PyMC3 model."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

import src.modeling.simulation_based_calibration_helpers as sbc
from src.exceptions import CacheDoesNotExistError, RequiredArgumentError
from src.loggers import logger
from src.managers.data_managers import DataFrameTransformation
from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.modeling import pymc3_sampling_api as pmapi
from src.project_enums import MockDataSize, ModelFitMethod, assert_never

ReplacementsDict = dict[TTShared, Union[pm.Minibatch, np.ndarray]]

ObservedVarName = str


class UnableToLocateNamedVariable(Exception):
    """Error when a named variable or object cannot be located."""

    pass


class SharedVariableDictionaryNotSet(AttributeError):
    """Error for when the shared variable dictionary should be available but is not."""

    pass


class PyMC3SamplingParameters(BaseModel):
    """Paramerers common to fitting a model in PyMC3."""

    draws: int = 1000
    prior_pred_samples: int = 1000


class MCMCSamplingParameters(PyMC3SamplingParameters):
    """Parameters for MCMC sampling."""

    tune: int = 2000
    cores: int = 4
    chains: int = 4
    init: str = "auto"
    n_init: int = 200000
    target_accept: float = 0.8  # default for pm.NUTS


class VISamplingParameters(PyMC3SamplingParameters):
    """Parameters for fitting by VI."""

    method: str = "advi"
    n_iterations: int = 50000


class SpecletModelDataManager(Protocol):
    """Protocol for the data manager of SpecletModel."""

    def get_data(self, read_kwargs: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Get data for modeling."""
        ...

    def set_data(self, data: pd.DataFrame, apply_transformations: bool = False) -> None:
        """Set data."""
        ...

    def add_transformation(self, fxn: DataFrameTransformation) -> None:
        """Add a transforming function."""
        ...

    def generate_mock_data(
        self, size: Union[MockDataSize, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data."""
        ...


class SpecletModel:
    """Base class to contain a PyMC3 model."""

    name: str
    _debug: bool
    cache_manager: Pymc3ModelCacheManager
    data_manager: SpecletModelDataManager

    model: Optional[pm.Model] = None
    observed_var_name: Optional[ObservedVarName] = None
    shared_vars: Optional[dict[str, TTShared]] = None
    advi_results: Optional[pmapi.ApproximationSamplingResults] = None
    mcmc_results: Optional[az.InferenceData] = None

    mcmc_sampling_params: MCMCSamplingParameters = MCMCSamplingParameters()
    advi_sampling_params: VISamplingParameters = VISamplingParameters()

    def __init__(
        self,
        name: str,
        data_manager: SpecletModelDataManager,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
    ) -> None:
        """Instantiate a Speclet Model.

        Args:
            name (str): Name of the model.
            root_cache_dir (Optional[Path], optional): Location for the cache directory.
              If None (default), then the project's default cache directory is used.
              Defaults to None.
            debug (bool, optional): Use debug mode? Defaults to False.
            data_manager (SpecletModelDataManager): Object that will manage the data.
        """
        self.name = name
        self._debug = debug
        self.cache_manager = Pymc3ModelCacheManager(
            name=name, root_cache_dir=root_cache_dir
        )
        self.data_manager = data_manager

    def __str__(self) -> str:
        """Describe the object.

        Returns:
            str: String description of the object.
        """
        msg = f"Speclet Model: '{self.name}'"
        if self.debug:
            msg += " (debug)"
        return msg

    @property
    def debug(self) -> bool:
        """Whether or not to use debug mode.

        Returns:
            bool: The current value for debug mode.
        """
        return self._debug

    @debug.setter
    def debug(self, new_value: bool) -> None:
        """Set the value for debug mode.

        This also changes the debug mode for the data manager. Changes are only made if
        the new value is different from the current value.

        Args:
            new_value (bool): New value for `debug`.
        """
        if new_value == self._debug:
            return
        logger.info(f"Changing value of debug to '{new_value}'.")
        self._debug = new_value

    def _reset_model_and_results(self, clear_cache: bool = False) -> None:
        logger.warning("Reseting all model and results.")
        self.model = None
        self.mcmc_results = None
        self.advi_results = None
        if clear_cache:
            self.clear_cache()

    def _get_batch_size(self) -> int:
        if self.debug:
            return 1000
        else:
            return 10000

    def clear_cache(self) -> None:
        """Clear all available caches for the model."""
        self.cache_manager.clear_all_caches()

    @abstractmethod
    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Define the PyMC3 model.

        This model must be overridden by an subclass to define the desired PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
            ObservedVarName: Name of the target variable in the model.
        """
        raise Exception(
            "The `model_specification()` method must be overridden by subclasses."
        )

    def build_model(self) -> None:
        """Build the PyMC3 model.

        Raises:
            AttributeError: Raised if there is no data manager.
            AttributeError: Raised the `model` attribute is still None after calling
              `self.model_specification()`
            AttributeError: Raised the `observed_var_name` attribute is still None
              after calling `self.model_specification()`
        """
        logger.debug("Building PyMC3 model.")
        logger.info("Calling `model_specification()` method.")
        self.model, self.observed_var_name = self.model_specification()

        if self.model is None:
            m = "The `model` attribute cannot be None at the end of the "
            m += "`build_model()` method."
            logger.error(m)
            raise AttributeError(m)

        if self.observed_var_name is None:
            m = "The `observed_var_name` attribute cannot be None at the end of the "
            m += "`build_model()` method."
            logger.error(m)
            raise AttributeError(m)

        return None

    def update_mcmc_sampling_parameters(self) -> None:
        """Override if MCMC sampling parameters need to be adjusted."""
        return None

    def mcmc_sample_model(
        self,
        draws: Optional[int] = None,
        tune: Optional[int] = None,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        target_accept: Optional[float] = None,
        prior_pred_samples: Optional[int] = None,
        random_seed: Optional[int] = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
        ignore_cache: bool = False,
    ) -> az.InferenceData:
        """MCMC sample the model.

        This method primarily wraps the `pymc3_sampling_api.pymc3_sampling_procedure()`
        function.

        Many of the key arguments default to None in the function call, but are replaced
        by the values in the `self.mcmc_sampling_params` attribute.

        Args:
            mcmc_draws (Optional[int], optional): Number of MCMC draws. Defaults to
              None.
            tune (Optional[int], optional): Number of tuning steps. Defaults to None.
            chains (Optional[int], optional): Number of chains. Defaults to 3.
            cores (Optional[int], optional): Number of cores. Defaults to
              None.
            target_accept (Optional[float], optional): MCMC target acceptance. Defaults
              to None.
            prior_pred_samples (Optional[int], optional): Number of samples from the
              prior distributions. Defaults to None.
            random_seed (Optional[int], optional): The random seed for sampling.
            Defaults to None.
            sample_kwargs (dict[str, Any], optional): Kwargs for the sampling method.
            Defaults to {}.
            ignore_cache (bool, optional): Should any cached results be ignored?
              Defaults to False.

        Raises:
            AttributeError: Raised if the PyMC3 model does not yet exist.

        Returns:
            az.InferenceData: The results of MCMC sampling.
        """
        logger.debug("Beginning MCMC sampling method.")
        self.update_mcmc_sampling_parameters()
        if draws is None:
            draws = self.mcmc_sampling_params.draws
        if tune is None:
            tune = self.mcmc_sampling_params.tune
        if chains is None:
            chains = self.mcmc_sampling_params.chains
        if cores is None:
            cores = self.mcmc_sampling_params.cores
        if target_accept is None:
            target_accept = self.mcmc_sampling_params.target_accept
        if prior_pred_samples is None:
            prior_pred_samples = self.mcmc_sampling_params.prior_pred_samples

        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )

        if not ignore_cache and self.mcmc_results is not None:
            logger.info("Returning results from stored `mcmc_results` attribute.")
            return self.mcmc_results

        if not ignore_cache and self.cache_manager.mcmc_cache_exists():
            logger.info("Returning results from cache.")
            self.mcmc_results = self.cache_manager.get_mcmc_cache()
            return self.mcmc_results

        if sample_kwargs is None:
            sample_kwargs = {}
        sample_kwargs["target_accept"] = target_accept

        logger.info("Beginning MCMC sampling.")
        self.mcmc_results = pmapi.pymc3_sampling_procedure(
            model=self.model,
            mcmc_draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            prior_pred_samples=prior_pred_samples,
            random_seed=random_seed,
            sample_kwargs=sample_kwargs,
        )
        logger.info("Finished MCMC sampling - caching results.")
        self.write_mcmc_cache()
        return self.mcmc_results

    def get_replacement_parameters(self) -> Optional[ReplacementsDict]:
        """Create a dictionary of PyMC3 variables to replace for ADVI fitting.

        This method is useful if you can take advantage of creating MiniBatch
        variables and replaced them using SharedVariables in the model. If not changed,
        this method returns None and has no effect on ADVI.

        Returns:
            Optional[ReplacementsDict]: Dictionary of variable replacements.
        """
        return None

    def get_advi_callbacks(self) -> list[Any]:
        """Prepare a list of callbacks for ADVI fitting.

        This can be overridden by subclasses to apply custom callbacks or change the
        parameters of the CheckParametersConvergence callback.

        Returns:
            list[Any]: list of callbacks.
        """
        return [
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ]

    def update_advi_sampling_parameters(self) -> None:
        """Override if ADVI fitting parameters need to be adjusted."""
        return None

    def advi_sample_model(
        self,
        method: Optional[str] = None,
        n_iterations: Optional[int] = None,
        draws: Optional[int] = None,
        prior_pred_samples: Optional[int] = None,
        random_seed: Optional[int] = None,
        ignore_cache: bool = False,
    ) -> tuple[az.InferenceData, pm.Approximation]:
        """ADVI fit the model.

        This method primarily wraps the
          `pymc3_sampling_api.pymc3_advi_approximation_procedure()` function.

        Many of the key arguments default to None in the function call, but are replaced
        by the values in the `self.advi_sampling_params` attribute.

        Args:
            model (pm.Model): PyMC3 model.
            method (Optional[str], optional): VI method to use. Defaults to None.
            n_iterations (Optional[int]): Maximum number of fitting steps. Defaults to
              None.
            draws (Optional[int], optional): Number of MCMC samples to draw from the fit
              model. Defaults to None.
            prior_pred_samples (Optional[int], optional): Number of samples from the
              prior distributions. Defaults to None.
            callbacks (list[Callable], optional): list of fitting callbacks. Default is
              None.
            random_seed (Optional[int], optional): The random seed for sampling.
              Defaults to None.
            fit_kwargs (dict[str, Any], optional): Kwargs for the fitting method.
              Defaults to {}.

        Raises:
            AttributeError: Raised if the model does not yet exist.

        Returns:
            tuple[az.InferenceData, pm.Approximation]: The results of fitting the model
              and the approximation object.
        """
        logger.info("Beginning ADVI fitting method.")
        self.update_advi_sampling_parameters()
        if method is None:
            method = self.advi_sampling_params.method
        if n_iterations is None:
            n_iterations = self.advi_sampling_params.n_iterations
        if draws is None:
            draws = self.advi_sampling_params.draws
        if prior_pred_samples is None:
            prior_pred_samples = self.advi_sampling_params.prior_pred_samples

        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )

        fit_kwargs: dict[str, Any] = {}
        replacements = self.get_replacement_parameters()
        if replacements is not None:
            fit_kwargs["more_replacements"] = replacements

        if self.advi_results is not None:
            logger.info("Returning results from stored `advi_results` attribute.")
            return self.advi_results

        if not ignore_cache and self.cache_manager.advi_cache_exists():
            logger.info("Returning results from cache.")
            self.advi_results = self.cache_manager.get_advi_cache()
            return self.advi_results

        logger.info("Beginning ADVI fitting.")
        _advi_results = pmapi.pymc3_advi_approximation_procedure(
            model=self.model,
            method=method,
            n_iterations=n_iterations,
            draws=draws,
            prior_pred_samples=prior_pred_samples,
            callbacks=self.get_advi_callbacks(),
            random_seed=random_seed,
            fit_kwargs=fit_kwargs,
        )
        self.advi_results = (
            pmapi.convert_samples_to_arviz(self.model, _advi_results),
            _advi_results.approximation,
        )
        logger.info("Finished ADVI fitting - caching results.")
        self.write_advi_cache()
        return self.advi_results

    def generate_mock_data(
        self, size: MockDataSize, random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data.

        Args:
            size (MockDataSize): Size of the dataset to mock.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        logger.info("Creating new simulation data.")
        mock_data = self.data_manager.generate_mock_data(
            size=size, random_seed=random_seed
        )
        return mock_data

    def run_simulation_based_calibration(
        self,
        results_path: Path,
        fit_method: ModelFitMethod,
        mock_data: Optional[pd.DataFrame] = None,
        size: Optional[MockDataSize] = None,
        random_seed: Optional[int] = None,
        fit_kwargs: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Run a round of simulation-based calibration.

        Pre-generated mock data can be passed for use in the simulation through the
        `mock_data` argument or novel data can be generated by passing a preferred size
        through the `size` argument.

        Args:
            results_path (Path): Where to store the results.
            fit_method (ModelFitMethod): Which method to use for fitting.
            random_seed (Optional[int], optional): Random seed (for reproducibility).
              Defaults to None.
            mock_data (Optional[pd.DataFrame], optional): Mock data to use for the SBC.
              Defaults to None.
            size (Optional[MockDataSize], optional): Size of the dataset to mock.
              Defaults to None.
            fit_kwargs (Optional[dict[Any, Any]], optional): Keyword arguments to be
              passed to the fitting method. Default is None.

        Raises:
            RequiredArgumentError: If neither `mock_data` nor `size` are supplied.
        """
        if fit_kwargs is None:
            fit_kwargs = {}

        sbc_fm = sbc.SBCFileManager(dir=results_path)

        if mock_data is not None:
            logger.info("Setting provided mock data as data.")
            self.data_manager.set_data(mock_data, apply_transformations=True)
            mock_data = self.data_manager.get_data()
        elif size is not None:
            logger.info("Creating new simulation data.")
            mock_data = self.data_manager.generate_mock_data(
                size=size, random_seed=random_seed
            )
        else:
            raise RequiredArgumentError(
                "Either `mock_data` or `size` must be provided."
            )

        logger.info("Building model for SBC.")
        self.build_model()
        assert self.model is not None
        assert self.observed_var_name is not None

        logger.info("Sampling from the prior for mock values for SBC.")
        with self.model:
            priors = pm.sample_prior_predictive(samples=1, random_seed=random_seed)

        logger.info("Updating observed variable with generated data.")
        mock_data[self.observed_var_name] = priors.get(self.observed_var_name).flatten()
        self.data_manager.set_data(mock_data, apply_transformations=False)
        logger.info("Saving mock data to SBC cache.")
        sbc_fm.save_sbc_data(mock_data)

        # Update shared variable with adjusted observed data.
        logger.info("Updating observed value with prior-sampled values.")
        self.update_observed_data(mock_data[self.observed_var_name].values)

        logger.info(f"Fitting model to mock data using {fit_method.value}.")
        if fit_method is ModelFitMethod.ADVI:
            res, _ = self.advi_sample_model(random_seed=random_seed, **fit_kwargs)
        elif fit_method is ModelFitMethod.MCMC:
            res = self.mcmc_sample_model(random_seed=random_seed, **fit_kwargs)
        else:
            assert_never(fit_method)

        logger.info("Making posterior summary for the SBC.")
        posterior_summary = az.summary(res, fmt="wide", hdi_prob=0.89)
        assert isinstance(posterior_summary, pd.DataFrame)

        logger.info("Using a SBC file manager to save SBC results.")
        sbc_fm.save_sbc_results(
            priors=priors,
            inference_obj=res,
            posterior_summary=posterior_summary,
        )

    def get_sbc(
        self, results_path: Path
    ) -> tuple[pd.DataFrame, sbc.SBCResults, sbc.SBCFileManager]:
        """Retrieve the data and results of an SBC.

        Args:
            results_path (Path): Directory containing the SBC results.

        Raises:
            CacheDoesNotExistError: Raised if the cache does not exist.

        Returns:
            tuple[pd.DataFrame, sbc.SBCResults, sbc.SBCFileManager]: The simulated data,
            the SBC results, and the file manager for the SBC.
        """
        sbc_fm = sbc.SBCFileManager(results_path)

        # Checks that data and results exist.
        if not sbc_fm.simulation_data_exists():
            raise CacheDoesNotExistError(sbc_fm.sbc_data_path)
        if not sbc_fm.all_data_exists():
            raise CacheDoesNotExistError(sbc_fm.dir)

        simulated_data = sbc_fm.get_sbc_data()
        sbc_results = sbc_fm.get_sbc_results()
        self.data_manager.set_data(simulated_data)
        if self.model is None:
            self.build_model()
        return self.data_manager.get_data(), sbc_results, sbc_fm

    def write_mcmc_cache(self) -> None:
        """Cache the MCMC sampling results."""
        if self.mcmc_results is not None:
            self.cache_manager.write_mcmc_cache(self.mcmc_results)
        else:
            logger.warning("Did not cache MCMC samples because they do not exist.")

    def write_advi_cache(self) -> None:
        """Cache the ADVI sampling results."""
        if self.advi_results is not None:
            self.cache_manager.write_advi_cache(
                self.advi_results[0], self.advi_results[1]
            )
        else:
            logger.warning("Did not cache MCMC samples because they do not exist.")

    def load_mcmc_cache(self) -> az.InferenceData:
        """Load MCMC from cache.

        Sets the cached MCMC result as the instance's `mcmc_results` attribute, too.

        Raises:
            CacheDoesNotExistError: Raised if the cache does not exist.

        Returns:
            az.InferenceData: Cached MCMC results.
        """
        if self.cache_manager.mcmc_cache_exists():
            self.mcmc_results = self.cache_manager.get_mcmc_cache()
            return self.mcmc_results
        else:
            raise CacheDoesNotExistError(
                self.cache_manager.mcmc_cache_delegate.cache_dir
            )

    def load_advi_cache(self) -> tuple[az.InferenceData, pm.Approximation]:
        """Load ADVI from cache.

        Sets the cached ADVI result as the instance's `advi_results` attribute, too.

        Raises:
            CacheDoesNotExistError: Raised if the cache does not exist.

        Returns:
            tuple[az.InferenceData, pm.Approximation]: Cached ADVI results.
        """
        if self.cache_manager.advi_cache_exists():
            _advi_results = self.cache_manager.get_advi_cache()
            self.advi_results = _advi_results
            return _advi_results
        else:
            raise CacheDoesNotExistError(
                self.cache_manager.advi_cache_delegate.cache_dir
            )

    def update_observed_data(self, new_data: np.ndarray) -> None:
        """Update the values for the shared tensor for observed data.

        Args:
            new_data (np.ndarray): New data to set in the shared tensor.
        """
        if self.shared_vars is None:
            raise SharedVariableDictionaryNotSet(
                "Cannot locate shared variable dictionary."
            )
        _var_name = f"{self.observed_var_name}_shared"
        observed_var_shared = self.shared_vars.get(_var_name)
        if observed_var_shared is not None:
            observed_var_shared.set_value(new_data)
            logger.info(f"Setting new data for observed variable: '{_var_name}'.")
        else:
            msg = f"Unable to set new values for observed variable: '{_var_name}'."
            logger.error(msg)
            raise UnableToLocateNamedVariable(msg)

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        return None
