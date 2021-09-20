"""Simple functions for common interactions with PyMC3."""

from math import floor
from typing import Optional, Union

import arviz as az
import numpy as np
import pymc3 as pm
import xarray as xr
from pymc3.model import PyMC3Variable
from theano import tensor as tt

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


def hierarchical_normal(
    name: str,
    shape: Union[int, tuple[int, ...]],
    centered: bool = True,
    mu: Optional[PyMC3Variable] = None,
    mu_mean: float = 0.0,
    mu_sd: float = 2.5,
    sigma_sd: float = 2.5,
) -> PyMC3Variable:
    """Create a non-centered parameterized hierarchical variable.

    Args:
        name (str): Variable name.
        shape (Union[int, tuple[int, ...]]): Variable shape.
        centered (bool, optional): Centered or non-centered parameterization? Defaults
          to `True` (centered).
        mu (Optional[PyMC3Variable], optional): Optional pre-made hyper-distribution
          mean. Defaults to None.
        mu_mean (float, optional): Mean of the hyper-distribution mean hyperparameter.
          Defaults to 0.0.
        mu_sd (float, optional): Standard deviation of the hyper-distribution mean
          hyperparameter. Defaults to 2.5.
        sigma_sd (float, optional): Standard deviation of the hyper-distribution
          standard deviation hyperparameter. Defaults to 2.5.

    Returns:
        PyMC3Variable: The create variable.
    """
    if mu is None:
        mu = pm.Normal(f"μ_{name}", mu_mean, mu_sd)

    sigma = pm.HalfNormal(f"σ_{name}", sigma_sd)

    if centered:
        v = pm.Normal(name, mu, sigma, shape=shape)
    else:
        delta = pm.Normal(f"Δ_{name}", 0.0, 1.0, shape=shape)
        v = pm.Deterministic(name, mu + delta * sigma)

    return v


def hierarchical_normal_with_avg(
    name: str,
    avg_map: dict[str, Union[float, np.ndarray, tt.TensorConstant]],
    shape: Union[int, tuple[int, ...]],
    centered: bool = True,
    mu: Optional[PyMC3Variable] = None,
    mu_mean: float = 0.0,
    mu_sd: float = 2.5,
    sigma_sd: float = 2.5,
    gamma_mean: float = 0.0,
    gamma_sd: float = 2.5,
) -> PyMC3Variable:
    """Create a non-centered hierarchical variable with group-level mean predictors.

    In a hierarchical model, "when one or more predictors correlate with the group or
    unit effects, a key Gauss-Markov assumption is violated," and this may result in
    poor estimates of parameter uncertainty. This function implements
    `src.modeling.pymc3_helpers.hierarchical_normal()` except includes predictors for
    group-level means to overcome this problem. See *Fitting Multilevel Models When
    Predictors and GroupEffects Correlate* by Bafumi and Gelman for details.

    Args:
        name (str): Variable name.
        avg_map (dict[str, Union[float, np.ndarray]]): Map of other predictor names to
          their group-level means. See the example.
        shape (Union[int, tuple[int, ...]]): Variable shape.
        centered (bool, optional): Centered or non-centered parameterization? Defaults
          to `True` (centered).
        mu (Optional[PyMC3Variable], optional): Optional pre-made hyper-distribution
          mean. Defaults to None.
        mu_mean (float, optional): Mean of the hyper-distribution mean hyperparameter.
          Defaults to 0.0.
        mu_sd (float, optional): Standard deviation of the hyper-distribution mean
          hyperparameter. Defaults to 2.5.
        sigma_sd (float, optional): Standard deviation of the hyper-distribution
          standard deviation hyperparameter. Defaults to 2.5.
        gamma_mean (float, optional): Mean of the group-level predictor variables.
          Defaults to 0.0.
        gamma_sd (float, optional): Standard deviation of the group-level predictor
          variables. Defaults to 2.5.

    Returns:
        PyMC3Variable: The create variable.

    Example:
        Example from *Modeling Shark Attacks in Python with PyMC3* in "Mixed Effects."
        (https://austinrochford.com/posts/2021-06-27-sharks-pymc3.html#mixed-effects)
        >>> def standardize(x: np.ndarray) -> np.ndarray:
        ...     return (x - x.mean()) / x.std()
        >>>
        >>> x_pop_bar = tt.constant(
        ...     standardize(
        ...         attacks_df.assign(x_pop=x_pop.eval())
        ...         .groupby(level="State")["x_pop"]
        ...         .mean()
        ...         .values
        ...     )
        ... )
        >>>
        >>> with pm.Model() as mixed_model:
        ...     β0 = noncentered_normal_with_avg(
        ...         "β0", {"pop": x_pop_bar}, n_state
        ...     )
    """
    if mu is None:
        mu = pm.Normal(f"μ_{name}", mu_mean, mu_sd)

    avg_terms_sum = 0.0

    for term_name, x_bar in avg_map.items():
        _gamma = pm.Normal(f"γ_{name}_{term_name}_bar", gamma_mean, gamma_sd) * x_bar
        avg_terms_sum += _gamma

    return hierarchical_normal(
        name=name,
        shape=shape,
        centered=centered,
        mu=mu + avg_terms_sum,
        mu_mean=mu_mean,
        mu_sd=mu_sd,
        sigma_sd=sigma_sd,
    )
