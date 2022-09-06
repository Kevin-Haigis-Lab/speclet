"""Simple functions for common interactions with PyMC."""

import arviz as az
import pymc as pm
import xarray as xr
from pymc.model import Variable as PyMCVariable


def get_random_variable_names(m: pm.Model) -> set[str]:
    """Retrieve the names of the random variables in a model.

    Args:
        m (pm.Model): PyMC model.

    Returns:
        list[str]: A list of the random variable names.
    """
    return {v.name for v in m.basic_RVs}


def get_deterministic_variable_names(m: pm.Model) -> set[str]:
    """Retrieve the names of the deterministic variables in a model.

    Args:
        m (pm.Model): PyMC model.

    Returns:
        list[str]: A list of the deterministic variable names.
    """
    rvs = get_random_variable_names(m)
    unobs_vars = {v.name for v in m.unobserved_RVs}
    return unobs_vars.difference(rvs)


def get_variable_names(m: pm.Model) -> set[str]:
    """Get all variable names from a model.

    Args:
        m (pm.Model): PyMC model.

    Returns:
        list[str]: list of unique variable names.
    """
    rvs = get_random_variable_names(m)
    obs_vars = {v.name for v in m.unobserved_RVs}
    return rvs.union(obs_vars)


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


def thin(trace: az.InferenceData, by: int) -> az.InferenceData:
    """Thin a posterior trace."""
    thinned_trace = trace.sel(draw=slice(0, None, by))
    assert isinstance(thinned_trace, az.InferenceData)
    return thinned_trace


def thin_posterior(trace: az.InferenceData, by: int) -> az.InferenceData:
    """Thin posterior draws."""
    assert hasattr(trace, "posterior"), "No posterior data in trace."
    trace.posterior = trace.posterior.sel(draw=slice(0, None, by))
    return trace


def thin_posterior_predictive(trace: az.InferenceData, by: int) -> az.InferenceData:
    """Thin posterior predictive draws."""
    _msg = "No posterior predictive data in trace."
    assert hasattr(trace, "posterior_predictive"), _msg
    trace.posterior_predictive = trace.posterior_predictive.sel(draw=slice(0, None, by))
    return trace


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
    dims: tuple[str, ...],
    centered: bool = True,
    mu: float | PyMCVariable = 0.0,
    mu_param: tuple[float, float] | None = (0.0, 2.5),
    sigma: PyMCVariable | None = None,
    sigma_params: tuple[float, float] | None = (1.1, 0.5),
) -> PyMCVariable:
    """Create a non-centered parameterized hierarchical variable."""
    if mu is None:
        assert mu_param is not None
        mu = pm.Normal(f"mu_{name}", mu_param[0], mu_param[1])

    if sigma is None:
        assert sigma_params is not None
        sigma = pm.Gamma(f"sigma_{name}", sigma_params[0], sigma_params[1])

    if centered:
        v = pm.Normal(name, mu, sigma, dims=dims)
    else:
        delta = pm.Normal(f"delta_{name}", 0.0, 1.0, dims=dims)
        v = pm.Deterministic(name, mu + delta * sigma, dims=dims)

    return v


# def hierarchical_normal_with_avg(
#     name: str,
#     avg_map: dict[str, Union[float, np.ndarray, tt.TensorConstant]],
#     shape: Union[int, tuple[int, ...]],
#     centered: bool = True,
#     mu: Optional[PyMCVariable] = None,
#     mu_mean: float = 0.0,
#     mu_sd: float = 2.5,
#     sigma_sd: float = 2.5,
#     gamma_mean: float = 0.0,
#     gamma_sd: float = 2.5,
# ) -> PyMCVariable:
#     """Create a non-centered hierarchical variable with group-level mean predictors.

#     In a hierarchical model, "when one or more predictors correlate with the group or
#     unit effects, a key Gauss-Markov assumption is violated," and this may result in
#     poor estimates of parameter uncertainty. This function implements
#     `src.modeling.pymc_helpers.hierarchical_normal()` except includes predictors for
#     group-level means to overcome this problem. See *Fitting Multilevel Models When
#     Predictors and GroupEffects Correlate* by Bafumi and Gelman for details.

#     Args:
#         name (str): Variable name.
#         avg_map (dict[str, Union[float, np.ndarray]]): Map of other predictor names to
#           their group-level means. See the example.
#         shape (Union[int, tuple[int, ...]]): Variable shape.
#         centered (bool, optional): Centered or non-centered parameterization? Defaults
#           to `True` (centered).
#         mu (Optional[PyMCVariable], optional): Optional pre-made hyper-distribution
#           mean. Defaults to None.
#         mu_mean (float, optional): Mean of the hyper-distribution mean hyperparameter.
#           Defaults to 0.0.
#         mu_sd (float, optional): Standard deviation of the hyper-distribution mean
#           hyperparameter. Defaults to 2.5.
#         sigma_sd (float, optional): Standard deviation of the hyper-distribution
#           standard deviation hyperparameter. Defaults to 2.5.
#         gamma_mean (float, optional): Mean of the group-level predictor variables.
#           Defaults to 0.0.
#         gamma_sd (float, optional): Standard deviation of the group-level predictor
#           variables. Defaults to 2.5.

#     Returns:
#         PyMCVariable: The create variable.

#     Example:
#         Example from *Modeling Shark Attacks in Python with PyMC* in "Mixed Effects."
#         (https://austinrochford.com/posts/2021-06-27-sharks-pymc3.html#mixed-effects)
#         >>> def standardize(x: np.ndarray) -> np.ndarray:
#         ...     return (x - x.mean()) / x.std()
#         >>>
#         >>> x_pop_bar = tt.constant(
#         ...     standardize(
#         ...         attacks_df.assign(x_pop=x_pop.eval())
#         ...         .groupby(level="State")["x_pop"]
#         ...         .mean()
#         ...         .values
#         ...     )
#         ... )
#         >>>
#         >>> with pm.Model() as mixed_model:
#         ...     β0 = noncentered_normal_with_avg(
#         ...         "β0", {"pop": x_pop_bar}, n_state
#         ...     )
#     """
#     if mu is None:
#         mu = pm.Normal(f"μ_{name}", mu_mean, mu_sd)

#     avg_terms_sum = 0.0

#     for term_name, x_bar in avg_map.items():
#         _gamma = pm.Normal(f"γ_{name}_{term_name}_bar", gamma_mean, gamma_sd) * x_bar
#         avg_terms_sum += _gamma

#     return hierarchical_normal(
#         name=name,
#         shape=shape,
#         centered=centered,
#         mu=mu + avg_terms_sum,
#         mu_mean=mu_mean,
#         mu_sd=mu_sd,
#         sigma_sd=sigma_sd,
#     )
