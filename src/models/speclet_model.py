"""Base class to contain a PyMC3 model."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

import src.modeling.simulation_based_calibration_helpers as sbc
from src.loggers import logger
from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.managers.model_data_managers import DataManager
from src.modeling import pymc3_sampling_api as pmapi

ReplacementsDict = Dict[TTShared, Union[pm.Minibatch, np.ndarray]]


class PyMC3SamplingParameters(BaseModel):
    """Paramerers common to fitting a model in PyMC3."""

    draws: int = 1000
    prior_pred_samples: int = 1000
    post_pred_samples: Optional[int] = None


class MCMCSamplingParameters(PyMC3SamplingParameters):
    """Parameters for MCMC sampling."""

    tune: int = 2000
    cores: int = 4
    chains: int = 4
    init: str = "auto"
    n_init: int = 200000
    target_accept: float = 0.9


class VISamplingParameters(PyMC3SamplingParameters):
    """Parameters for fitting by VI."""

    method: str = "advi"
    n_iterations: int = 100000


class SpecletModel:
    """Base class to contain a PyMC3 model."""

    name: str
    _debug: bool
    cache_manager: Pymc3ModelCacheManager
    data_manager: Optional[DataManager] = None

    model: Optional[pm.Model] = None
    observed_var_name: Optional[str] = None
    shared_vars: Optional[Dict[str, TTShared]] = None
    advi_results: Optional[Tuple[az.InferenceData, pm.Approximation]] = None
    mcmc_results: Optional[az.InferenceData] = None

    mcmc_sampling_params: MCMCSamplingParameters = MCMCSamplingParameters()
    advi_sampling_params: VISamplingParameters = VISamplingParameters()

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
    ) -> None:
        """Instantiate a Speclet Model.

        Args:
            name (str): Name of the model.
            root_cache_dir (Optional[Path], optional): Location for the cache directory.
              If None (default), then the project's default cache directory is used.
              Defaults to None.
            debug (bool, optional): Use debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. Defaults to None.
        """
        self.name = name
        self._debug = debug
        self.cache_manager = Pymc3ModelCacheManager(
            name=name, root_cache_dir=root_cache_dir
        )
        if data_manager is not None:
            self.data_manager = data_manager
            self.data_manager.debug = self._debug

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
        if self.data_manager is not None:
            self.data_manager.debug = new_value

    def _reset_model_and_results(self):
        logger.warning("Reseting all model and results.")
        self.model = None
        self.mcmc_results = None
        self.advi_results = None
        self.cache_manager.clear_all_caches()

    @abstractmethod
    def model_specification(self) -> Tuple[pm.Model, str]:
        """Define the PyMC3 model.

        This model must be overridden by an subclass to define the desired PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
            str: Name of the target variable in the model.
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
        if self.data_manager is None:
            logger.error("Cannot build a model without a data manager.")
            raise AttributeError("Cannot build a model without a data manager.")

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
        mcmc_draws: Optional[int] = None,
        tune: Optional[int] = None,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        target_accept: Optional[float] = None,
        prior_pred_samples: Optional[int] = None,
        post_pred_samples: Optional[int] = None,
        random_seed: Optional[int] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
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
            post_pred_samples (Optional[int], optional): Number of samples for posterior
              predictions.
            Defaults to None.
            random_seed (Optional[int], optional): The random seed for sampling.
            Defaults to None.
            sample_kwargs (Dict[str, Any], optional): Kwargs for the sampling method.
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
        if mcmc_draws is None:
            mcmc_draws = self.mcmc_sampling_params.draws
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
        if post_pred_samples is None:
            post_pred_samples = self.mcmc_sampling_params.post_pred_samples

        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )

        if self.mcmc_results is not None:
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
        _mcmc_results = pmapi.pymc3_sampling_procedure(
            model=self.model,
            mcmc_draws=mcmc_draws,
            tune=tune,
            chains=chains,
            cores=cores,
            prior_pred_samples=prior_pred_samples,
            post_pred_samples=post_pred_samples,
            random_seed=random_seed,
            sample_kwargs=sample_kwargs,
        )
        self.mcmc_results = pmapi.convert_samples_to_arviz(self.model, _mcmc_results)
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

    def get_advi_callbacks(self) -> List[Any]:
        """Prepare a list of callbacks for ADVI fitting.

        This can be overridden by subclasses to apply custom callbacks or change the
        parameters of the CheckParametersConvergence callback.

        Returns:
            List[Any]: List of callbacks.
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
        post_pred_samples: Optional[int] = None,
        random_seed: Optional[int] = None,
        ignore_cache: bool = False,
    ) -> Tuple[az.InferenceData, pm.Approximation]:
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
            post_pred_samples (Optional[int], optional): Number of samples for posterior
              predictions. Defaults to None.
            callbacks (List[Callable], optional): List of fitting callbacks. Default is
              None.
            random_seed (Optional[int], optional): The random seed for sampling.
              Defaults to None.
            fit_kwargs (Dict[str, Any], optional): Kwargs for the fitting method.
              Defaults to {}.

        Raises:
            AttributeError: Raised if the model does not yet exist.

        Returns:
            Tuple[az.InferenceData, pm.Approximation]: The results of fitting the model
              and the approximation object.
        """
        logger.debug("Beginning ADVI fitting method.")
        self.update_advi_sampling_parameters()
        if method is None:
            method = self.advi_sampling_params.method
        if n_iterations is None:
            n_iterations = self.advi_sampling_params.n_iterations
        if draws is None:
            draws = self.advi_sampling_params.draws
        if prior_pred_samples is None:
            prior_pred_samples = self.advi_sampling_params.prior_pred_samples
        if post_pred_samples is None:
            post_pred_samples = self.advi_sampling_params.post_pred_samples

        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )

        fit_kwargs: Dict[str, Any] = {}
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
            post_pred_samples=post_pred_samples,
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

    def run_simulation_based_calibration(
        self,
        results_path: Path,
        random_seed: Optional[int] = None,
        size: str = "large",
        fit_kwargs: Dict[Any, Any] = {},
    ) -> None:
        """Run a round of simulation-based calibration.

        Args:
            results_path (Path): Where to store the results.
            random_seed (Optional[int], optional): Random seed (for reproducibility).
              Defaults to None.
            size (str, optional): Size of the data set to mock. Defaults to "large".
            fit_kwargs (Dict[Any, Any], optional): Keyword arguments to be passed to the
              fitting method. Default is an empty dictionary.
        """
        assert self.data_manager is not None
        mock_data = self.data_manager.generate_mock_data(
            size=size, random_seed=random_seed
        )

        self.build_model()
        assert self.model is not None
        assert self.observed_var_name is not None

        with self.model:
            priors = pm.sample_prior_predictive(samples=1, random_seed=random_seed)

        mock_data[self.observed_var_name] = priors.get(self.observed_var_name).flatten()
        self.data = mock_data

        res, _ = self.advi_sample_model(random_seed=random_seed, **fit_kwargs)
        posterior_summary = az.summary(res, fmt="wide", hdi_prob=0.89)
        assert isinstance(posterior_summary, pd.DataFrame)
        assert isinstance(posterior_summary, pd.DataFrame)
        results_manager = sbc.SBCFileManager(dir=results_path)
        results_manager.save_sbc_results(
            priors=priors,
            inference_obj=res,
            posterior_summary=posterior_summary,
        )

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
