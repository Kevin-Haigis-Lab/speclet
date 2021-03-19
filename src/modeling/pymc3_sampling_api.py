#!/usr/bin/env python3

"""Standardization of the interactions with PyMC3 sampling."""

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import arviz as az
import pretty_errors
import pymc3 as pm
from colorama import Back, Fore, Style, init

init(autoreset=True)

default_cache_dir = Path("pymc3_model_cache")


#### ---- Cache management  ----- ####


def write_pickle(x: Any, fp: Path) -> None:
    """Write `x` to disk as a pickle file.

    Args:
        x (Any): The object to pickle.
        fp (Path): File path of the pickle.

    Returns:
        [None]: None
    """
    with open(fp, "wb") as f:
        pickle.dump(x, f)
    return None


def get_pickle(fp: Path) -> Any:
    """Read a pickled file into Python.

    Args:
        fp (Path): The pickle file path.

    Returns:
        Any: The pickled object.
    """
    with open(fp, "rb") as f:
        d = pickle.load(f)
    return d


# TODO: I should create a class to hold these data. --> ArviZ.InferenceData?


def cache_file_names(cache_dir: Path) -> Tuple[Path, Path, Path]:
    """Generate standard caching file names.

    Args:
        cache_dir (Path): The cache directory.

    Returns:
        Tuple[Path, Path, Path]: A tuple of file paths.
    """
    post_file_path = cache_dir / "posterior-predictive-check.pkl"
    prior_file_path = cache_dir / "prior-predictive-check.pkl"
    approx_file_path = cache_dir / "vi-approximation.pkl"
    return prior_file_path, post_file_path, approx_file_path


def package_cached_sampling_data(
    trace: pm.backends.base.MultiTrace,
    post_check: Dict[str, Any],
    prior_check: Dict[str, Any],
) -> Dict[str, Any]:
    """Organize the results of sampling for caching.

    Args:
        trace (pm.backends.base.MultiTrace): PyMC3 sampling trace.
        post_check (Dict[str, Any]): PyMC3 posterior predictions.
        prior_check (Dict[str, Any]): PyMC3 prior predictions.

    Returns:
        Dict[str, Any]: A dictionary containing the results of sampling in PyMC3.
    """
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
    """Organize the results of VI for caching.

    Args:
        approx (pm.MeanField): PyMC3 VI approximation.
        trace (pm.backends.base.MultiTrace): PyMC3 sampling trace.
        post_check (Dict[str, Any]): PyMC3 posterior predictions.
        prior_check (Dict[str, Any]): PyMC3 prior predictions.

    Returns:
        Dict[str, Any]: A dictionary containing the results of sampling in PyMC3.
    """
    return {
        "approximation": approx,
        "trace": trace,
        "posterior_predictive": post_check,
        "prior_predictive": prior_check,
    }


def read_cached_sampling(cache_dir: Path, model: pm.Model) -> Dict[str, Any]:
    """Read sampling from cache.

    Args:
        cache_dir (Path): The cache directory.
        model (pm.Model): The model corresponding to the cached sampling.

    Returns:
        Dict[str, Any]: A dictionary containing the cached results.
    """
    prior_file_path, post_file_path, _ = cache_file_names(cache_dir)

    trace = pm.load_trace(cache_dir.as_posix(), model=model)
    post_check = get_pickle(post_file_path)
    prior_check = get_pickle(prior_file_path)

    return package_cached_sampling_data(trace, post_check, prior_check)


def read_cached_vi(cache_dir: Path, draws: int = 1000) -> Dict[str, Any]:
    """Read VI from cache.

    Args:
        cache_dir (Path): The cache directory.
        model (pm.Model): The model corresponding to the cached VI.

    Returns:
        Dict[str, Any]: A dictionary containing the cached results.
    """
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
    """Run a standard PyMC3 sampling procedure.

    Args:
        model (pm.Model): PyMC3 model.
        num_mcmc (int, optional): Number of MCMC draws. Defaults to 1000.
        tune (int, optional): Number of tuning steps. Defaults to 1000.
        chains (int, optional): Number of chains. Defaults to 2.
        cores (Optional[int], optional): Number of cores. Defaults to None.
        prior_check_samples (int, optional): Number of samples from the prior distributions. Defaults to 1000.
        ppc_samples (int, optional): Number of samples for posterior predictions. Defaults to 1000.
        random_seed (Optional[int], optional): The random seed for sampling. Defaults to None.
        cache_dir (Optional[Path], optional): A directory to cache results. Defaults to None.
        force (bool, optional): Should the model be fit even if there is an existing cache? Defaults to False.
        sample_kwargs (Dict[str, Any], optional): Kwargs for the sampling method. Defaults to {}.

    Returns:
        Dict[str, Any]: A dictionary containing the sampling results.
    """
    if cache_dir is not None:
        prior_file_path, post_file_path, _ = cache_file_names(cache_dir)

    if not force and cache_dir is not None and cache_dir.exists():
        return read_cached_sampling(cache_dir, model=model)

    with model:
        prior_check = pm.sample_prior_predictive(
            prior_check_samples, random_seed=random_seed
        )
        trace = pm.sample(
            draws=num_mcmc,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            **sample_kwargs,
        )
        post_check = pm.sample_posterior_predictive(
            trace, samples=ppc_samples, random_seed=random_seed
        )
    if cache_dir is not None:
        pm.save_trace(trace, directory=cache_dir.as_posix(), overwrite=True)
        write_pickle(post_check, post_file_path)
        write_pickle(prior_check, prior_file_path)

    return package_cached_sampling_data(trace, post_check, prior_check)


def pymc3_advi_approximation_procedure(
    model: pm.Model,
    method: str = "advi",
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
    """Run a standard PyMC3 ADVI fitting procedure.

    Args:
        model (pm.Model): PyMC3 model.
        method (str): VI method to use. Defaults to "advi".
        n_iterations (int): Maximum number of fitting steps. Defaults to 100000.
        draws (int, optional): Number of MCMC samples to draw from the fit model. Defaults to 1000.
        prior_check_samples (int, optional): Number of samples from the prior distributions. Defaults to 1000.
        post_check_samples (int, optional): Number of samples for posterior predictions. Defaults to 1000.
        callbacks (List[Callable], optional): List of fitting callbacks. Default is None.
        random_seed (Optional[int], optional): The random seed for sampling. Defaults to None.
        cache_dir (Optional[Path], optional): A directory to cache results. Defaults to None.
        force (bool, optional): Should the model be fit even if there is an existing cache? Defaults to False.
        fit_kwargs (Dict[str, Any], optional): Kwargs for the fitting method. Defaults to {}.

    Returns:
        Dict[str, Any]: A dictionary containing the fitting and sampling results.
    """
    if cache_dir is not None:
        prior_file_path, post_file_path, approx_file_path = cache_file_names(cache_dir)

    if not force and cache_dir is not None and cache_dir.exists():
        try:
            return read_cached_vi(cache_dir)
        except Exception as err:
            print(f"ERROR: {err}")

    with model:
        prior_check = pm.sample_prior_predictive(
            prior_check_samples, random_seed=random_seed
        )
        approx = pm.fit(n_iterations, method=method, callbacks=callbacks, **fit_kwargs)
        advi_trace = approx.sample(draws)
        post_check = pm.sample_posterior_predictive(
            trace=advi_trace, samples=post_check_samples, random_seed=random_seed
        )

    if cache_dir is not None:
        pm.save_trace(advi_trace, directory=cache_dir.as_posix(), overwrite=True)
        write_pickle(post_check, post_file_path)
        write_pickle(prior_check, prior_file_path)
        write_pickle(approx, approx_file_path)

    return package_cached_vi_data(
        approx=approx, trace=advi_trace, post_check=post_check, prior_check=prior_check
    )


def samples_to_arviz(model: pm.Model, res: Dict[str, Any]) -> az.InferenceData:
    """Turn the results of `pymc3_sampling_procedure()` into a standard ArviZ object.

    Args:
        model (pm.Model): The model used for the
        res (Dict[str, Any]): The results of the fitting process.

    Returns:
        az.InferenceData: A standard ArviZ data object.
    """
    return az.from_pymc3(
        trace=res["trace"],
        model=model,
        prior=res["prior_predictive"],
        posterior_predictive=res["posterior_predictive"],
    )
