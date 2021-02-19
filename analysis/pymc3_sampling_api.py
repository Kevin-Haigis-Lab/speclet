import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
from colorama import Back, Fore, Style, init

init(autoreset=True)

default_cache_dir = Path("pymc3_model_cache")


#### ---- Dialogue ---- ####


def info(m: str):
    print(Fore.BLACK + Style.DIM + m)
    return None


#### ---- Cache management  ----- ####


def write_pickle(x: Any, fp: Path) -> None:
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


def get_pickle(fp: Path) -> Any:
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


def cache_file_names(cache_dir: Path) -> Tuple[Path, Path, Path]:
    post_file_path = cache_dir / "posterior-predictive-check.pkl"
    prior_file_path = cache_dir / "prior-predictive-check.pkl"
    approx_file_path = cache_dir / "vi-approximation.pkl"
    return prior_file_path, post_file_path, approx_file_path


def package_cached_sampling_data(
    trace: pm.backends.base.MultiTrace,
    post_check: Dict[str, Any],
    prior_check: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "trace": trace,
        "posterior_predictive": post_check,
        "prior_predictive": prior_check,
    }


def package_cached_vi_data(
    approx: pm.MeanField,
    trace: pm.backends.base.MultiTrace,
    post_check: Dict[str, Any],
    prior_check: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "approximation": approx,
        "trace": trace,
        "posterior_predictive": post_check,
        "prior_predictive": prior_check,
    }


def read_cached_sampling(cache_dir: Path, model: pm.Model) -> Dict[str, Any]:
    info("Loading cached trace and posterior sample...")
    prior_file_path, post_file_path, _ = cache_file_names(cache_dir)

    trace = pm.load_trace(cache_dir.as_posix(), model=model)
    post_check = get_pickle(post_file_path)
    prior_check = get_pickle(prior_file_path)

    return package_cached_sampling_data(trace, post_check, prior_check)


def read_cached_vi(cache_dir: Path, draws: int = 1000) -> Dict[str, Any]:
    info("Loading cached trace and posterior sample...")
    prior_file_path, post_file_path, approx_file_path = cache_file_names(cache_dir)

    post_check = get_pickle(post_file_path)
    prior_check = get_pickle(prior_file_path)
    approx = get_pickle(approx_file_path)
    trace = approx.sample(draws)

    return package_cached_vi_data(
        approx=approx, trace=trace, post_check=post_check, prior_check=prior_check
    )


#### ---- Interface with PyMC3 ---- ####


def pymc3_sampling_procedure(
    model: pm.Model,
    num_mcmc: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    cores: Optional[int] = None,
    prior_check_samples: int = 1000,
    ppc_samples: int = 1000,
    random_seed: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    sample_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
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
        prior_file_path, post_file_path, _ = cache_file_names(cache_dir)

    if not force and cache_dir is not None and cache_dir.exists():
        return read_cached_sampling(cache_dir, model=model)

    with model:
        info("Sampling from prior distributions.")
        prior_check = pm.sample_prior_predictive(
            prior_check_samples, random_seed=random_seed
        )
        info("Sampling from posterior.")
        trace = pm.sample(
            draws=num_mcmc,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            **sample_kwargs
        )
        info("Posterior predicitons.")
        post_check = pm.sample_posterior_predictive(
            trace, samples=ppc_samples, random_seed=random_seed
        )
    if cache_dir is not None:
        info("Caching trace and posterior sample...")
        pm.save_trace(trace, directory=cache_dir.as_posix(), overwrite=True)
        write_pickle(post_check, post_file_path)
        write_pickle(prior_check, prior_file_path)

    return package_cached_sampling_data(trace, post_check, prior_check)


def pymc3_advi_approximation_procedure(
    model: pm.Model,
    n_iterations: int = 100000,
    draws: int = 1000,
    prior_check_samples: int = 1000,
    post_check_samples: int = 1000,
    callbacks: Optional[List[Callable]] = None,
    random_seed: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    fit_kwargs: Dict[Any, Any] = {},
) -> Dict[str, Any]:
    if cache_dir is not None:
        prior_file_path, post_file_path, approx_file_path = cache_file_names(cache_dir)

    if not force and cache_dir is not None and cache_dir.exists():
        return read_cached_vi(cache_dir)

    with model:
        info("Sampling from prior distributions.")
        prior_check = pm.sample_prior_predictive(
            prior_check_samples, random_seed=random_seed
        )
        info("Running ADVI approximation.")
        approx = pm.fit(n_iterations, method="advi", callbacks=callbacks, **fit_kwargs)
        info("Sampling from posterior.")
        advi_trace = approx.sample(draws)
        info("Posterior predicitons.")
        post_check = pm.sample_posterior_predictive(
            trace=advi_trace, samples=post_check_samples, random_seed=random_seed
        )

    if cache_dir is not None:
        info("Caching trace and posterior sample...")
        pm.save_trace(advi_trace, directory=cache_dir.as_posix(), overwrite=True)
        write_pickle(post_check, post_file_path)
        write_pickle(prior_check, prior_file_path)
        write_pickle(approx, approx_file_path)

    return package_cached_vi_data(
        approx=approx, trace=advi_trace, post_check=post_check, prior_check=prior_check
    )


def samples_to_arviz(model: pm.Model, res: Dict[str, Any]) -> az.InferenceData:
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
