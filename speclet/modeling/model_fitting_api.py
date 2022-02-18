"""Standardization of the interactions with model sampling."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import arviz as az
import pandas as pd
import pymc as pm
from pydantic import BaseModel
from stan.model import Model as StanModel

from speclet.bayesian_models import BayesianModelProtocol
from speclet.loggers import logger
from speclet.modeling.custom_pymc_callbacks import ProgressPrinterCallback
from speclet.modeling.fitting_arguments import (
    ModelingSamplingArguments,
    PymcFitArguments,
    PymcSampleArguments,
    StanMCMCSamplingArguments,
)
from speclet.project_configuration import on_hms_cluster
from speclet.project_enums import ModelFitMethod, assert_never
from speclet.utils.general import resolve_optional_kwargs


def _get_kwargs_dict(
    data: Optional[Union[BaseModel, dict[str, Any]]]
) -> dict[str, Any]:
    if data is None:
        return {}
    elif isinstance(data, BaseModel):
        return data.dict()
    else:
        return data


# ---- Result Types ----


@dataclass
class ApproximationSamplingResults:
    """The results of ADVI fitting and sampling."""

    inference_data: az.InferenceData
    approximation: pm.Approximation


# ---- Interface with PyMC ----


def _update_return_inferencedata_kwarg(
    sampling_kwargs: Optional[PymcSampleArguments],
) -> Optional[PymcSampleArguments]:
    if sampling_kwargs is None:
        return sampling_kwargs

    if not sampling_kwargs.return_inferencedata:
        logger.warning("Switching `return_inferencedata` to `True`.")
        sampling_kwargs.return_inferencedata = True

    return sampling_kwargs


def _specific_o2_progress(sampling_kwargs: dict[str, Any]) -> None:
    if "callback" in sampling_kwargs:
        return
    if not on_hms_cluster():
        return
    sampling_kwargs["callback"] = ProgressPrinterCallback(every_n=5)
    sampling_kwargs["progressbar"] = False
    return


def _sample_prior(
    model: pm.Model, prior_pred_samples: Optional[int], trace: az.InferenceData
) -> None:
    if prior_pred_samples is not None and prior_pred_samples > 0:
        with model:
            prior_pred = pm.sample_prior_predictive(
                samples=prior_pred_samples, return_inferencedata=True
            )
            assert isinstance(prior_pred, az.InferenceData)
            trace.extend(prior_pred)
    else:
        logger.info("Not sampling from prior predictive.")
    return None


def fit_pymc_mcmc(
    model: pm.Model,
    prior_pred_samples: Optional[int] = None,
    sampling_kwargs: Optional[Union[PymcSampleArguments, dict[str, Any]]] = None,
) -> az.InferenceData:
    """Run a standardized PyMC sampling procedure.

    The value for `return_inferencedata` will be set to `True`.

    Args:
        model (pm.Model): PyMC model.
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
    _specific_o2_progress(kwargs)
    with model:
        trace = pm.sample(**kwargs)
        _ = pm.sample_posterior_predictive(trace=trace, extend_inferencedata=True)

    assert isinstance(trace, az.InferenceData)
    _sample_prior(model, prior_pred_samples, trace)
    return trace


def fit_pymc_vi(
    model: pm.Model,
    prior_pred_samples: Optional[int] = None,
    fit_kwargs: Optional[Union[PymcFitArguments, dict[str, Any]]] = None,
) -> ApproximationSamplingResults:
    """Run a standard PyMC ADVI fitting procedure.

    TODO: update for v4

    Args:
        model (pm.Model): PyMC model.
        prior_pred_samples (int, optional): Number of samples from the prior
        distributions. Defaults to 1000. If less than 1, no prior samples are taken.
        fit_kwargs (Optional[Pymc3FitArguments], optional): Keyword arguments for the
        fit method. Defaults to `None`.

    Returns:
        ApproximationSamplingResults: A collection of the fitting and sampling results.
    """
    kwargs = _get_kwargs_dict(fit_kwargs)
    draws = kwargs.pop("draws", PymcFitArguments().draws)

    with model:
        approx = pm.fit(**kwargs)
        trace = az.from_pymc3(trace=approx.sample(draws))
        _ = pm.sample_posterior_predictive(trace=trace, extend_inferencedata=True)

    assert isinstance(approx, pm.Approximation)
    assert isinstance(trace, az.InferenceData)
    _sample_prior(model, prior_pred_samples, trace)
    return ApproximationSamplingResults(inference_data=trace, approximation=approx)


# --- Interface with Stan ---


def fit_stan_mcmc(
    stan_model: StanModel,
    sampling_kwargs: Optional[Union[StanMCMCSamplingArguments, dict[str, Any]]] = None,
    az_kwargs: Optional[dict[str, Any]] = None,
) -> az.InferenceData:
    """Fit a Stan model.

    Args:
        stan_model (StanModel): The Stan model to fit.
        sampling_kwargs (Optional[StanMCMCSamplingArguments], optional): Optional
        fitting keyword arguments. Defaults to None

    Returns:
        az.InferenceData: Model posterior draws.
    """
    az_kwargs = resolve_optional_kwargs(az_kwargs)
    kwargs = _get_kwargs_dict(sampling_kwargs)
    _ = kwargs.pop("random_seed", None)  # remove 'random_seed'
    post = stan_model.sample(**kwargs)
    return az.from_pystan(posterior=post, posterior_model=stan_model, **az_kwargs)


# ---- Dispatching ----


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
        posterior = fit_stan_mcmc(
            stan_model,
            sampling_kwargs=kwargs,
            az_kwargs=model.stan_idata_addons(data=data),
        )
        return posterior
    elif fit_method is ModelFitMethod.PYMC_MCMC:
        if sampling_kwargs is not None:
            kwargs = sampling_kwargs.pymc_mcmc
        else:
            kwargs = None
        pymc_model = model.pymc_model(data=data)
        return fit_pymc_mcmc(
            model=pymc_model, prior_pred_samples=0, sampling_kwargs=kwargs
        )
    elif fit_method is ModelFitMethod.PYMC_ADVI:
        if sampling_kwargs is not None:
            kwargs = sampling_kwargs.pymc_advi
        else:
            kwargs = None
        pymc_model = model.pymc_model(data=data)
        return fit_pymc_vi(
            model=pymc_model, prior_pred_samples=0, fit_kwargs=kwargs
        ).inference_data
    else:
        assert_never(fit_method)
