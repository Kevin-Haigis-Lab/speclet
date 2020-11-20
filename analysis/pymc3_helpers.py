import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm


def write_pickle(x, fp):
    """
    Write `x` to disk as a pickle file.

    Parameters
    ----------
    x: obj
        Object to be pickled
    fp: pathlib.Path
        Path for where to write the pickle
    """
    with open(fp, "wb") as f:
        pickle.dump(x, f)
    return None


def get_pickle(fp):
    """
    Read a pickled file into Python.

    Parameters
    ----------
    fp: pathlib.Path
        Path for where to get the pickle
    """
    with open(fp, "rb") as f:
        d = pickle.load(f)
    return d


def pymc3_sampling_procedure(
    model,
    num_mcmc=1000,
    tune=1000,
    chains=2,
    cores=None,
    prior_check_samples=1000,
    ppc_samples=1000,
    random_seed=1234,
    cache_dir=None,
    force=False,
    sample_kwags={},
):
    """
    Run the standard PyMC3 sampling procedure.

    Parameters
    ----------
    model: pymc3.Model
        A model from PyMC3
    num_mcmc: int
        Number of MCMC samples
    tune: int
        Number of MCMC tuning steps
    chains: int
        Number of of MCMC chains
    cores: int
        Number of cores for MCMC
    prior_check_samples: int
        Number of prior predictive samples to take
    ppc_samples: int
        Number of posterior predictive samples to take
    random_seed: int
        Random seed to use in all sampling processes
    cache_dir: pathlib.Path
        The directory to cache the output (leave as `None` to skip caching)
    force: bool
        Ignore cached results and compute trace and predictive checks,
    sample_kwags: dict
        Keyword arguments passed to `pm.sample()`.

    Returns
    -------
    dict
        Contains the "trace", "posterior_predictive", and "prior_predictive"
    """
    if cache_dir is not None:
        post_file_path = cache_dir / "posterior-predictive-check.pkl"
        prior_file_path = cache_dir / "prior-predictive-check.pkl"

    if not force and cache_dir is not None and cache_dir.exists():
        print("Loading cached trace and posterior sample...")
        trace = pm.load_trace(cache_dir.as_posix(), model=model)
        post_check = get_pickle(post_file_path)
        prior_check = get_pickle(prior_file_path)
    else:
        with model:
            prior_check = pm.sample_prior_predictive(
                prior_check_samples, random_seed=random_seed
            )
            trace = pm.sample(
                num_mcmc,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                **sample_kwags
            )
            post_check = pm.sample_posterior_predictive(
                trace, samples=ppc_samples, random_seed=random_seed
            )
        if cache_dir is not None:
            print("Caching trace and posterior sample...")
            pm.save_trace(trace, directory=cache_dir.as_posix(), overwrite=True)
            write_pickle(post_check, post_file_path)
            write_pickle(prior_check, prior_file_path)
    return {
        "trace": trace,
        "posterior_predictive": post_check,
        "prior_predictive": prior_check,
    }


def samples_to_arviz(model, res):
    """
    Turn the results of `pymc3_sampling_procedure()` into a standard ArviZ object.

    Parameters
    ----------
    model: pymc3.Model
        The model
    res: dict
        The sampling results from `pymc3_sampling_procedure()`

    Returns
    -------
    arviz.InferenceData
    """
    return az.from_pymc3(
        trace=res["trace"],
        model=model,
        prior=res["prior_predictive"],
        posterior_predictive=res["posterior_predictive"],
    )
