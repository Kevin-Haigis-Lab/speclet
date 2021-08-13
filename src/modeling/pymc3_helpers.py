"""Simple functions for common interactions with PyMC3."""

from math import floor
from typing import Optional

import arviz as az
import pymc3 as pm
import xarray as xr

from src.exceptions import RequiredArgumentError


def _rm_log(model_vars: list[str]) -> list[str]:
    return [v.replace("_log__", "") for v in model_vars]


def get_random_variable_names(m: pm.Model, rm_log: bool = False) -> list[str]:
    """Retrieve the names of the random variables in a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffixes be removed? Defaults to
          False.

    Returns:
        list[str]: A list of the random variable names.
    """
    model_vars: list[str] = [v.name for v in m.free_RVs]
    if rm_log:
        model_vars = _rm_log(model_vars)
    return model_vars


def get_deterministic_variable_names(m: pm.Model, rm_log: bool = False) -> list[str]:
    """Retrieve the names of the deterministic variables in a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffixes be removed? Defaults to
          False.

    Returns:
        list[str]: A list of the deterministic variable names.
    """
    model_vars = list(
        set([v.name for v in m.unobserved_RVs])  # noqa: C403
        .difference(get_random_variable_names(m))
        .difference(get_random_variable_names(m, rm_log=True))
    )
    if rm_log:
        model_vars = _rm_log(model_vars)
    return model_vars


def get_variable_names(m: pm.Model, rm_log: bool = False) -> list[str]:
    """Get all variable names from a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffices be removed? Defaults to
          False.

    Returns:
        list[str]: list of unique variable names.
    """
    rvs = [v.name for v in m.unobserved_RVs] + [v.name for v in m.observed_RVs]
    if rm_log:
        rvs = _rm_log(rvs)
    return list(set(rvs))


def get_posterior_names(data: az.InferenceData) -> list[str]:
    """Get the names of variables present in the posterior of a model.

    Args:
        data (az.InferenceData): The ArviZ data object with sampled posterior.

    Raises:
        KeyError: Raised if the posterior data is not present.

    Returns:
        list[str]: A list of the vairables with their posteriors sampled.
    """
    if "posterior" in data:
        return [str(i) for i in data["posterior"].data_vars]
    else:
        raise KeyError("Input ArviZ InferenceData object does not have posterior data.")


def thin_posterior(
    posterior: xr.DataArray,
    thin_to: Optional[int] = None,
    step_size: Optional[int] = None,
) -> xr.DataArray:
    """Thin a posterior to a specific number of values or by ever n steps.

    Args:
        posterior (xr.DataArray): Posterior to thin.
        thin_to (int, optional): Number of samples to thin the posterior down to.
          Defaults to None.
        step_size (Optional[int], optional): Size of the step for the thinning process.
          Defaults to None.

    Returns:
        xr.DataArray: The thinned posterior distribution.
    """
    if thin_to is not None:
        step_size = floor(posterior.shape[1] / thin_to)
    elif step_size is None:
        raise RequiredArgumentError(
            "A value must be passed to either `thin_to` or `step_size`."
        )
    return posterior.sel(draw=slice(0, None, step_size))


def get_one_chain(posterior: xr.DataArray, chain_num: int = 0) -> xr.DataArray:
    """Extract just the first chain from a posterior.

    Args:
        posterior (xr.DataArray): The posterior.
        chain_num (int, optional): The index of the chain to keep. Defaults to 0.

    Returns:
        xr.DataArray: The same posterior data structure but with only the desired chain.
    """
    return posterior.sel(chain=chain_num)
