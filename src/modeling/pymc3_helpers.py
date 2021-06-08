"""Simple functions for common interactions with PyMC3."""

from typing import List

import pymc3 as pm


def _rm_log(model_vars: List[str]) -> List[str]:
    return [v.replace("_log__", "") for v in model_vars]


def get_random_variable_names(m: pm.Model, rm_log: bool = False) -> List[str]:
    """Retrieve the names of the random variables in a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffixes be removed? Defaults to
          False.

    Returns:
        List[str]: A list of the random variable names.
    """
    model_vars: List[str] = [v.name for v in m.free_RVs]
    if rm_log:
        model_vars = _rm_log(model_vars)
    return model_vars


def get_deterministic_variable_names(m: pm.Model, rm_log: bool = False) -> List[str]:
    """Retrieve the names of the deterministic variables in a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffixes be removed? Defaults to
          False.

    Returns:
        List[str]: A list of the deterministic variable names.
    """
    model_vars = list(
        set([v.name for v in m.unobserved_RVs])
        .difference(get_random_variable_names(m))
        .difference(get_random_variable_names(m, rm_log=True))
    )
    if rm_log:
        model_vars = _rm_log(model_vars)
    return model_vars


def get_variable_names(m: pm.Model, rm_log: bool = False) -> List[str]:
    """Get all variable names from a model.

    Args:
        m (pm.Model): PyMC3 model.
        rm_log (bool, optional): Should the '_log__' suffices be removed? Defaults to
          False.

    Returns:
        List[str]: List of unique variable names.
    """
    rvs = [v.name for v in m.unobserved_RVs] + [v.name for v in m.observed_RVs]
    if rm_log:
        rvs = _rm_log(rvs)
    return list(set(rvs))
