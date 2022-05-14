"""Standardization of the interactions with model sampling."""

from dataclasses import dataclass
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm
import pymc.sampling_jax
from pydantic import BaseModel

from speclet.bayesian_models import BayesianModelProtocol
from speclet.loggers import logger
from speclet.modeling.custom_pymc_callbacks import ProgressPrinterCallback
from speclet.modeling.fitting_arguments import (
    ModelingSamplingArguments,
    PymcFitArguments,
    PymcSampleArguments,
)
from speclet.modeling.fitting_arguments import (
    PymcSamplingNumpyroArguments as PymcSamplingNumpyroArgs,
)
from speclet.project_configuration import on_hms_cluster
from speclet.project_enums import ModelFitMethod, assert_never


def _get_kwargs_dict(data: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
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


def _specific_o2_progress(sampling_kwargs: dict[str, Any]) -> None:
    if "callback" in sampling_kwargs:
        return
    if not on_hms_cluster():
        return
    sampling_kwargs["callback"] = ProgressPrinterCallback(every_n=5)
    sampling_kwargs["progressbar"] = False
    return


def _sample_prior(
    model: pm.Model, prior_pred_samples: int | None, trace: az.InferenceData
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
    prior_pred_samples: int | None = None,
    sampling_kwargs: PymcSampleArguments | dict[str, Any] | None = None,
) -> az.InferenceData:
    """Run a standardized PyMC sampling procedure.

    The value for `return_inferencedata` will be set to `True`.

    Args:
        model (pm.Model): PyMC model.
        prior_pred_samples (int | None, optional): Number of samples from the prior
        distributions. Defaults to None. If `None` or less than 1, no prior samples
        are taken.
        sample_kwargs (PymcSampleArguments | dict[str, Any] | None, optional): Keyword
        arguments for the sampling method. Defaults to `None`.

    Returns:
        az.InferenceData: Model posterior sample.
    """
    kwargs = _get_kwargs_dict(sampling_kwargs)
    _specific_o2_progress(kwargs)
    with model:
        trace = pm.sample(**kwargs)
        _ = pm.sample_posterior_predictive(trace=trace, extend_inferencedata=True)

    assert isinstance(trace, az.InferenceData)
    _sample_prior(model, prior_pred_samples, trace)
    return trace


def fit_pymc_mcmc_numpyro(
    model: pm.Model,
    prior_pred_samples: int | None = None,
    sampling_kwargs: PymcSamplingNumpyroArgs | dict[str, Any] | None = None,
) -> az.InferenceData:
    """Run a standardized PyMC sampling procedure using the Numpyro JAX backend.

    The value for `return_inferencedata` will be set to `True`.

    Args:
        model (pm.Model): PyMC model.
        prior_pred_samples (int | None, optional): Number of samples from the prior
        distributions. Defaults to None. If `None` or less than 1, no prior samples
        are taken.
        sample_kwargs (PymcSamplingNumpyroArgs | dict[str, Any] | None): Keyword
        arguments for the sampling method. Defaults to `None`.

    Returns:
        az.InferenceData: Model posterior sample.
    """
    kwargs = _get_kwargs_dict(sampling_kwargs)
    with model:
        trace = pymc.sampling_jax.sample_numpyro_nuts(**kwargs)
        _ = pm.sample_posterior_predictive(trace=trace, extend_inferencedata=True)

    assert isinstance(trace, az.InferenceData)
    _sample_prior(model, prior_pred_samples, trace)
    return trace


def fit_pymc_vi(
    model: pm.Model,
    prior_pred_samples: int | None = None,
    fit_kwargs: PymcFitArguments | dict[str, Any] | None = None,
) -> ApproximationSamplingResults:
    """Run a standard PyMC ADVI fitting procedure.

    TODO: update for v4 (not ready yet)

    Args:
        model (pm.Model): PyMC model.
        prior_pred_samples (int, optional): Number of samples from the prior
        distributions. Defaults to 1000. If less than 1, no prior samples are taken.
        fit_kwargs (PymcFitArguments | dict[str, Any] | None, optional): Keyword
        arguments for the fit method. Defaults to `None`.

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


# ---- Dispatching ----


def fit_model(
    model: BayesianModelProtocol,
    data: pd.DataFrame,
    fit_method: ModelFitMethod,
    sampling_kwargs: ModelingSamplingArguments | None = None,
    seed: int | None = None,
) -> az.InferenceData:
    """Fit a model using a specified method.

    Args:
        model (BayesianModelProtocol): Bayesian model to fit.
        data (pd.DataFrame): CRISPR screen data to use.
        fit_method (ModelFitMethod): Fitting method.
        sampling_kwargs (ModelingSamplingArguments | None, optional): Optional sampling
        keyword arguments. Defaults to `None`.
        seed (int | None, optional): Random seed for models. Defaults to `None`.

    Returns:
        az.InferenceData: Model posterior.
    """
    pymc_model = model.pymc_model(data=data, seed=seed)
    if fit_method is ModelFitMethod.PYMC_MCMC:
        kwargs = sampling_kwargs.pymc_mcmc if sampling_kwargs is not None else None
        return fit_pymc_mcmc(
            model=pymc_model, prior_pred_samples=0, sampling_kwargs=kwargs
        )
    elif fit_method is ModelFitMethod.PYMC_NUMPYRO:
        kwargs = sampling_kwargs.pymc_numpyro if sampling_kwargs is not None else None
        return fit_pymc_mcmc_numpyro(
            model=pymc_model, prior_pred_samples=0, sampling_kwargs=kwargs
        )
    elif fit_method is ModelFitMethod.PYMC_ADVI:
        kwargs = sampling_kwargs.pymc_advi if sampling_kwargs is not None else None
        return fit_pymc_vi(
            model=pymc_model, prior_pred_samples=0, fit_kwargs=kwargs
        ).inference_data
    else:
        assert_never(fit_method)
