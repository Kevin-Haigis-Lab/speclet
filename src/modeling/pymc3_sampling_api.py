#!/usr/bin/env python3

"""Standardization of the interactions with PyMC3 sampling."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import arviz as az
import numpy as np
import pymc3 as pm

from src.loggers import logger

#### ---- Result Types ---- ####


@dataclass
class ApproximationSamplingResults:
    """The results of ADVI fitting and sampling."""

    inference_data: az.InferenceData
    approximation: pm.Approximation


#### ---- Interface with PyMC3 ---- ####


def _extend_trace_with_prior_and_posterior(
    trace: az.InferenceData,
    prior: Optional[dict[str, np.ndarray]] = None,
    post: Optional[dict[str, np.ndarray]] = None,
) -> None:
    if prior is not None:
        trace.extend(az.from_pymc3(prior=prior))
    if post is not None:
        trace.extend(az.from_pymc3(posterior_predictive=post))
    return None


def pymc3_sampling_procedure(
    model: pm.Model,
    prior_pred_samples: Optional[int] = 500,
    random_seed: Optional[int] = None,
    sample_kwargs: Optional[dict[str, Any]] = None,
) -> az.InferenceData:
    """Run a standardized PyMC3 sampling procedure.

    Args:
        model (pm.Model): PyMC3 model.
        prior_pred_samples (Optional[int], optional): Number of samples from the prior
          distributions. Defaults to 1000. If `None` or less than 1, no prior samples
          are taken.
        random_seed (Optional[int], optional): The random seed for sampling.
          Defaults to `None`.
        sample_kwargs (Dict[str, Any], optional): Kwargs for the sampling method.
          Defaults to `None`.

    Returns:
        az.InferenceData: ArviZ standardized data set.
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    with model:
        trace = pm.sample(
            random_seed=random_seed, return_inferencedata=True, **sample_kwargs
        )
        post_pred = pm.sample_posterior_predictive(trace, random_seed=random_seed)

    assert isinstance(trace, az.InferenceData)

    prior_pred: Optional[dict[str, np.ndarray]] = None
    if prior_pred_samples is not None and prior_pred_samples > 0:
        with model:
            prior_pred = pm.sample_prior_predictive(
                prior_pred_samples, random_seed=random_seed
            )
    else:
        logger.info("Not sampling from prior predictive.")

    with model:
        _extend_trace_with_prior_and_posterior(trace, prior=prior_pred, post=post_pred)
    return trace


def pymc3_advi_approximation_procedure(
    model: pm.Model,
    method: str = "advi",
    n_iterations: int = 100000,
    draws: int = 1000,
    prior_pred_samples: Optional[int] = 500,
    callbacks: Optional[list[Callable]] = None,
    random_seed: Optional[int] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> ApproximationSamplingResults:
    """Run a standard PyMC3 ADVI fitting procedure.

    TODO (@jhrcook): Change `method` from a string to a literal and get supported
    options from PyMC3.

    Args:
        model (pm.Model): PyMC3 model.
        method (str): VI method to use. Defaults to "advi".
        n_iterations (int): Maximum number of fitting steps. Defaults to 100000.
        draws (int, optional): Number of MCMC samples to draw from the fit model.
          Defaults to 1000.
        prior_pred_samples (int, optional): Number of samples from the prior
          distributions. Defaults to 1000. If less than 1, no prior samples are taken.
        callbacks (List[Callable], optional): List of fitting callbacks.
          Default is None.
        random_seed (Optional[int], optional): The random seed for sampling.
          Defaults to None.
        fit_kwargs (Dict[str, Any], optional): Kwargs for the fitting method.
          Defaults to None.

    Returns:
        ApproximationSamplingResults: A collection of the fitting and sampling results.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    with model:
        approx = pm.fit(n_iterations, method=method, callbacks=callbacks, **fit_kwargs)
        trace = az.from_pymc3(trace=approx.sample(draws))
        post_pred = pm.sample_posterior_predictive(trace=trace, random_seed=random_seed)

    assert isinstance(trace, az.InferenceData)

    prior_pred: Optional[dict[str, np.ndarray]] = None
    if prior_pred_samples is not None and prior_pred_samples > 0:
        with model:
            prior_pred = pm.sample_prior_predictive(
                prior_pred_samples, random_seed=random_seed
            )
    else:
        logger.info("Not sampling from prior predictive.")

    _extend_trace_with_prior_and_posterior(trace, prior=prior_pred, post=post_pred)
    return ApproximationSamplingResults(inference_data=trace, approximation=approx)
