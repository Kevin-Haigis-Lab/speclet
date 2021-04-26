"""Base class to contain a PyMC3 model."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

import src.modeling.simulation_based_calibration_helpers as sbc
from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.managers.model_data_managers import DataManager
from src.modeling import pymc3_sampling_api as pmapi

ReplacementsDict = Dict[TTShared, Union[pm.Minibatch, np.ndarray]]


class SpecletModel:
    """Base class to contain a PyMC3 model."""

    name: str
    _debug: bool
    cache_manager: Pymc3ModelCacheManager
    data_manager: Optional[DataManager] = None

    model: Optional[pm.Model] = None
    observed_var_name: Optional[str] = None
    shared_vars: Optional[Dict[str, TTShared]] = None
    advi_results: Optional[pmapi.ApproximationSamplingResults] = None
    mcmc_results: Optional[pmapi.MCMCSamplingResults] = None

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
        print(f"current value of self._debug: {self._debug}")
        if new_value == self._debug:
            return
        self._debug = new_value
        if self.data_manager is not None:
            self.data_manager.debug = new_value

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
        if self.data_manager is None:
            raise AttributeError("Cannot build a model without a data manager.")

        self.model, self.observed_var_name = self.model_specification()

        if self.model is None:
            m = "The `model` attribute cannot be None at the end of the "
            m += "`build_model()` method."
            raise AttributeError(m)

        if self.observed_var_name is None:
            m = "The `observed_var_name` attribute cannot be None at the end of the "
            m += "`build_model()` method."
            raise AttributeError(m)

        return None

    def mcmc_sample_model(
        self,
        mcmc_draws: int = 1000,
        tune: int = 1000,
        chains: int = 3,
        cores: Optional[int] = None,
        prior_pred_samples: int = 1000,
        post_pred_samples: int = 1000,
        random_seed: Optional[int] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        ignore_cache: bool = False,
    ) -> pmapi.MCMCSamplingResults:
        """MCMC sample the model.

        This method primarily wraps the `pymc3_sampling_api.pymc3_sampling_procedure()`
        function.

        Args:
            mcmc_draws (int, optional): Number of MCMC draws. Defaults to 1000.
            tune (int, optional): Number of tuning steps. Defaults to 1000.
            chains (int, optional): Number of chains. Defaults to 3.
            cores (Optional[int], optional): Number of cores. Defaults to None.
            prior_pred_samples (int, optional): Number of samples from the prior
            distributions. Defaults to 1000.
            post_pred_samples (int, optional): Number of samples for posterior
              predictions.
            Defaults to 1000.
            random_seed (Optional[int], optional): The random seed for sampling.
            Defaults to None.
            sample_kwargs (Dict[str, Any], optional): Kwargs for the sampling method.
            Defaults to {}.
            ignore_cache (bool, optional): Should any cahced results be ignored?
              Defaults to False.

        Raises:
            AttributeError: Raised if the PyMC3 model does not yet exist.

        Returns:
            pmapi.MCMCSamplingResults: The results of MCMC sampling.
        """
        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )

        if self.mcmc_results is not None:
            return self.mcmc_results

        if not ignore_cache and self.cache_manager.mcmc_cache_exists():
            self.mcmc_results = self.cache_manager.get_mcmc_cache(model=self.model)
            return self.mcmc_results

        self.mcmc_results = pmapi.pymc3_sampling_procedure(
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
        self.cache_manager.write_mcmc_cache(self.mcmc_results)
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

        This can be overriden by subclasses to apply custom callbacks or change the
        parameters of the CheckParametersConvergence callback.

        Returns:
            List[Any]: List of callbacks.
        """
        return [
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ]

    def advi_sample_model(
        self,
        method: str = "advi",
        n_iterations: int = 100000,
        draws: int = 1000,
        prior_pred_samples: int = 1000,
        post_pred_samples: int = 1000,
        random_seed: Optional[int] = None,
        ignore_cache: bool = False,
    ) -> pmapi.ApproximationSamplingResults:
        """ADVI fit the model.

        Args:
            model (pm.Model): PyMC3 model.
            method (str): VI method to use. Defaults to "advi".
            n_iterations (int): Maximum number of fitting steps. Defaults to 100000.
            draws (int, optional): Number of MCMC samples to draw from the fit model.
            Defaults to 1000.
            prior_pred_samples (int, optional): Number of samples from the prior
            distributions. Defaults to 1000.
            post_pred_samples (int, optional): Number of samples for posterior
              predictions.
            Defaults to 1000.
            callbacks (List[Callable], optional): List of fitting callbacks.
            Default is None.
            random_seed (Optional[int], optional): The random seed for sampling.
            Defaults to None.
            fit_kwargs (Dict[str, Any], optional): Kwargs for the fitting method.
            Defaults to {}.

        Raises:
            AttributeError: Raised if the model does not yet exist.

        Returns:
            ApproximationSamplingResults: A collection of the fitting and sampling
              results.
        """
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
            return self.advi_results

        if not ignore_cache and self.cache_manager.advi_cache_exists():
            self.advi_results = self.cache_manager.get_advi_cache()
            return self.advi_results

        self.advi_results = pmapi.pymc3_advi_approximation_procedure(
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
        self.cache_manager.write_advi_cache(self.advi_results)
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

        res = self.advi_sample_model(random_seed=random_seed, **fit_kwargs)
        posterior_summary = az.summary(res.trace, fmt="wide", hdi_prob=0.89)
        assert isinstance(posterior_summary, pd.DataFrame)
        assert isinstance(posterior_summary, pd.DataFrame)
        az_results = pmapi.convert_samples_to_arviz(model=self.model, res=res)
        results_manager = sbc.SBCFileManager(dir=results_path)
        results_manager.save_sbc_results(
            priors=priors,
            inference_obj=az_results,
            posterior_summary=posterior_summary,
        )
