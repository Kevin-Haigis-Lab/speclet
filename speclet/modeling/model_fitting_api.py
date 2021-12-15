"""Standardization of the interactions with model sampling."""

from dataclasses import dataclass
from typing import Any, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from stan.model import Model as StanModel

from speclet.bayesian_models import BayesianModelProtocol
from speclet.loggers import logger
from speclet.modeling.fitting_arguments import (
    ModelingSamplingArguments,
    Pymc3FitArguments,
    Pymc3SampleArguments,
    StanMCMCSamplingArguments,
)
from speclet.project_enums import ModelFitMethod, assert_never


def _get_kwargs_dict(data: Optional[BaseModel]) -> dict[str, Any]:
    if data is None:
        return {}
    return data.dict()


# ---- Result Types ---- #


@dataclass
class ApproximationSamplingResults:
    """The results of ADVI fitting and sampling."""

    inference_data: az.InferenceData
    approximation: pm.Approximation


# ---- Interface with PyMC3 ---- #


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


def _update_return_inferencedata_kwarg(
    sampling_kwargs: Optional[Pymc3SampleArguments],
) -> Optional[Pymc3SampleArguments]:
    if sampling_kwargs is None:
        return sampling_kwargs

    if not sampling_kwargs.return_inferencedata:
        logger.warning("Switching `return_inferencedata` to `True`.")
        sampling_kwargs.return_inferencedata = True

    return sampling_kwargs


def fit_pymc3_mcmc(
    model: pm.Model,
    prior_pred_samples: Optional[int] = None,
    sampling_kwargs: Optional[Pymc3SampleArguments] = None,
) -> az.InferenceData:
    """Run a standardized PyMC3 sampling procedure.

    Args:
        model (pm.Model): PyMC3 model.
        prior_pred_samples (Optional[int], optional): Number of samples from the prior
        distributions. Defaults to None. If `None` or less than 1, no prior samples
        are taken.
        sample_kwargs (Dict[str, Any], optional): Keyword arguments for the sampling
        method. Defaults to `None`.

    Returns:
        az.InferenceData: Model posterior sample.
    """
    sampling_kwargs = _update_return_inferencedata_kwarg(sampling_kwargs)
    kwargs = _get_kwargs_dict(sampling_kwargs)
    random_seed = kwargs.get("random_seed", None)

    with model:
        trace = pm.sample(**kwargs)
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


def fit_pymc3_vi(
    model: pm.Model,
    prior_pred_samples: Optional[int] = None,
    fit_kwargs: Optional[Pymc3FitArguments] = None,
) -> ApproximationSamplingResults:
    """Run a standard PyMC3 ADVI fitting procedure.

    Args:
        model (pm.Model): PyMC3 model.
        prior_pred_samples (int, optional): Number of samples from the prior
        distributions. Defaults to 1000. If less than 1, no prior samples are taken.
        fit_kwargs (Optional[Pymc3FitArguments], optional): Keyword arguments for the
        fit method. Defaults to `None`.

    Returns:
        ApproximationSamplingResults: A collection of the fitting and sampling results.
    """
    kwargs = _get_kwargs_dict(fit_kwargs)
    random_seed = kwargs.get("random_seed", None)
    draws = kwargs.pop("draws", Pymc3FitArguments().draws)

    with model:
        approx = pm.fit(**kwargs)
        trace = az.from_pymc3(trace=approx.sample(draws))
        post_pred = pm.sample_posterior_predictive(trace=trace, random_seed=random_seed)

    assert isinstance(approx, pm.Approximation)
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

    return ApproximationSamplingResults(inference_data=trace, approximation=approx)


# --- Interface with Stan --- #


def fit_stan_mcmc(
    stan_model: StanModel, sampling_kwargs: Optional[StanMCMCSamplingArguments] = None
) -> az.InferenceData:
    """Fit a Stan model.

    Args:
        stan_model (StanModel): The Stan model to fit.
        sampling_kwargs (Optional[StanMCMCSamplingArguments], optional): Optional
        fitting keyword arguments. Defaults to None

    Returns:
        az.InferenceData: Model posterior draws.
    """
    kwargs = _get_kwargs_dict(sampling_kwargs)
    _ = kwargs.pop("random_seed", None)  # remove 'random_seed'
    post = stan_model.sample(**kwargs)

    # TODO: add in data for posterior predictive, coordinates, etc.

    return az.from_pystan(posterior=post, posterior_model=stan_model)


# ---- Dispatching ---- #


def fit_model(
    model: BayesianModelProtocol,
    data: pd.DataFrame,
    fit_method: ModelFitMethod,
    sampling_kwargs: Optional[ModelingSamplingArguments] = None,
) -> az.InferenceData:
    """Fit a model using a specified method.

    Args:
        model (BayesianModelProtocol): Bayesian model to fit.
        data (pd.DataFrame): CRISPR screen data to use.
        fit_method (ModelFitMethod): Fitting method.
        sampling_kwargs (Optional[ModelingSamplingArguments], optional): Optional
        sampling keyword arguments. Defaults to None.

    Returns:
        az.InferenceData: Model posterior.
    """
    if fit_method is ModelFitMethod.STAN_MCMC:
        if sampling_kwargs is not None:
            kwargs = sampling_kwargs.stan_mcmc
        else:
            kwargs = None
        seed = None if kwargs is None else kwargs.random_seed
        stan_model = model.stan_model(data=data, random_seed=seed)
        return fit_stan_mcmc(stan_model, sampling_kwargs=kwargs)
    elif fit_method is ModelFitMethod.PYMC3_MCMC:
        if sampling_kwargs is not None:
            kwargs = sampling_kwargs.pymc3_mcmc
        else:
            kwargs = None
        pymc3_model = model.pymc3_model(data=data)
        return fit_pymc3_mcmc(
            model=pymc3_model, prior_pred_samples=0, sampling_kwargs=kwargs
        )
    elif fit_method is ModelFitMethod.PYMC3_ADVI:
        if sampling_kwargs is not None:
            kwargs = sampling_kwargs.pymc3_advi
        else:
            kwargs = None
        pymc3_model = model.pymc3_model(data=data)
        return fit_pymc3_vi(
            model=pymc3_model, prior_pred_samples=0, fit_kwargs=kwargs
        ).inference_data
    else:
        assert_never(fit_method)
