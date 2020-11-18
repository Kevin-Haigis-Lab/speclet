import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
from pathlib import Path


def write_pickle(x, fp):
    """Write `x` to disk as a pickle file."""
    with open(fp, "wb") as f:
        pickle.dump(x, f)
    return None


def get_pickle(fp):
    """Read a pickled file into Python."""
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
    sample_kwags=None,
):
    """
    Run the standard PyMC3 sampling procedure.

        Parameters:
            model(pymc3 model): A model from PyMC3
            num_mcmc(int): number of MCMC samples
            tune(int): number of MCMC tuning steps
            chains(int): number of of MCMC chains
            cores(int): number of cores for MCMC
            prior_check_samples(int): number of prior predictive samples to take
            ppc_samples(int): number of posterior predictive samples to take
            random_seed(int): random seed to use in all sampling processes
            cache_dir(Path): the directory to cache the output (leave as `None` to skip caching)
            force(bool): ignore cached results and compute trace and predictive checks,
            sample_kwags(dict): keyword arguments passed to `pm.sample()`.
        Returns:
            dict: contains the "trace", "posterior_predictive", and "prior_predictive"
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
