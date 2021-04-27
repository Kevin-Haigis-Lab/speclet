#!/usr/bin/env python3

"""Standardization of the interactions with PyMC3 sampling."""

from typing import Any, Callable, Dict, List, Optional, Union

import arviz as az
import numpy as np
import pymc3 as pm
from pydantic import BaseModel, validator

#### ---- Result Types ---- ####


class MCMCSamplingResults(BaseModel):
    """The results of MCMC sampling."""

    trace: pm.backends.base.MultiTrace
    prior_predictive: Dict[str, np.ndarray]
    posterior_predictive: Dict[str, np.ndarray]

    class Config:
        """Configuration for pydantic validation."""

        arbitrary_types_allowed = True

    @validator("trace")
    def validate_trace(cls, trace):
        """Validate a PyMC3 MultiTrace object.

        Args:
            trace ([type]): MultiTrace object.

        Raises:
            ValueError: If the object does not satisfy pre-determined requirements.

        Returns:
            [type]: The original object (if valid).
        """
        trace_methods = dir(trace)
        expected_methods = ["get_values"]
        for method in expected_methods:
            if method not in trace_methods:
                raise ValueError(
                    f"Object passed for trace does not have the method '{method}'."
                )
        return trace


class ApproximationSamplingResults(MCMCSamplingResults):
    """The results of ADVI fitting and sampling."""

    approximation: pm.Approximation


def convert_samples_to_arviz(
    model: pm.Model,
    res: Union[MCMCSamplingResults, ApproximationSamplingResults],
) -> az.InferenceData:
    """Turn the results from a sampling procedure into a standard ArviZ object.

    Args:
        model (pm.Model): The PyMC3 model.
        res (Union[MCMCSamplingResults, ApproximationSamplingResults]): The results of
          the sampling/fitting process.

    Returns:
        az.InferenceData: A standard ArviZ data object.
    """
    return az.from_pymc3(
        trace=res.trace,
        model=model,
        prior=res.prior_predictive,
        posterior_predictive=res.posterior_predictive,
    )


#### ---- Interface with PyMC3 ---- ####


def pymc3_sampling_procedure(
    model: pm.Model,
    mcmc_draws: int = 1000,
    tune: int = 1000,
    chains: int = 3,
    cores: Optional[int] = None,
    prior_pred_samples: int = 1000,
    post_pred_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
    sample_kwargs: Optional[Dict[str, Any]] = None,
) -> MCMCSamplingResults:
    """Run a standard PyMC3 sampling procedure.

    Args:
        model (pm.Model): PyMC3 model.
        mcmc_draws (int, optional): Number of MCMC draws. Defaults to 1000.
        tune (int, optional): Number of tuning steps. Defaults to 1000.
        chains (int, optional): Number of chains. Defaults to 3.
        cores (Optional[int], optional): Number of cores. Defaults to None.
        prior_pred_samples (int, optional): Number of samples from the prior
          distributions. Defaults to 1000.
        post_pred_samples (Optional[int], optional): Number of samples for posterior
          predictions. The default behavior (None) to keep the same size as the MCMC
          trace. Defaults to None.
        random_seed (Optional[int], optional): The random seed for sampling.
          Defaults to None.
        sample_kwargs (Dict[str, Any], optional): Kwargs for the sampling method.
          Defaults to {}.

    Returns:
        MCMCSamplingResults: A collection of the fitting and sampling results.
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    keep_size = True if post_pred_samples is None else None

    with model:
        prior_pred = pm.sample_prior_predictive(
            prior_pred_samples, random_seed=random_seed
        )
        trace = pm.sample(
            draws=mcmc_draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            **sample_kwargs,
        )

        post_pred = pm.sample_posterior_predictive(
            trace,
            samples=post_pred_samples,
            keep_size=keep_size,
            random_seed=random_seed,
        )

    return MCMCSamplingResults(
        trace=trace,
        prior_predictive=prior_pred,
        posterior_predictive=post_pred,
    )


def pymc3_advi_approximation_procedure(
    model: pm.Model,
    method: str = "advi",
    n_iterations: int = 100000,
    draws: int = 1000,
    prior_pred_samples: int = 1000,
    post_pred_samples: int = 1000,
    callbacks: Optional[List[Callable]] = None,
    random_seed: Optional[int] = None,
    fit_kwargs: Optional[Dict[Any, Any]] = None,
) -> ApproximationSamplingResults:
    """Run a standard PyMC3 ADVI fitting procedure.

    Args:
        model (pm.Model): PyMC3 model.
        method (str): VI method to use. Defaults to "advi".
        n_iterations (int): Maximum number of fitting steps. Defaults to 100000.
        draws (int, optional): Number of MCMC samples to draw from the fit model.
          Defaults to 1000.
        prior_pred_samples (int, optional): Number of samples from the prior
          distributions. Defaults to 1000.
        post_pred_samples (int, optional): Number of samples for posterior predictions.
          Defaults to 1000.
        callbacks (List[Callable], optional): List of fitting callbacks.
          Default is None.
        random_seed (Optional[int], optional): The random seed for sampling.
          Defaults to None.
        fit_kwargs (Dict[str, Any], optional): Kwargs for the fitting method.
          Defaults to {}.

    Returns:
        ApproximationSamplingResults: A collection of the fitting and sampling results.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    with model:
        prior_pred = pm.sample_prior_predictive(
            prior_pred_samples, random_seed=random_seed
        )
        approx = pm.fit(n_iterations, method=method, callbacks=callbacks, **fit_kwargs)
        advi_trace = approx.sample(draws)
        post_pred = pm.sample_posterior_predictive(
            trace=advi_trace, samples=post_pred_samples, random_seed=random_seed
        )

    return ApproximationSamplingResults(
        trace=advi_trace,
        prior_predictive=prior_pred,
        posterior_predictive=post_pred,
        approximation=approx,
    )
