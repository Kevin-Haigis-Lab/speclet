#!/usr/bin/env python3

"""Foundational model object and related functions for the Speclet project."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import arviz as az
import numpy as np
import pretty_errors
import pymc3 as pm
from pydantic import BaseModel, validator


class ModelCachePaths(BaseModel):
    """Paths for model caching."""

    trace_path: Path
    prior_predictive_path: Path
    posterior_predictive_path: Path
    approximation_path: Path


class MCMCSamplingResults(BaseModel):
    """The results of MCMC sampling."""

    trace: pm.backends.base.MultiTrace
    prior_predictive: Dict[str, np.ndarray]
    posterior_predictive: Dict[str, np.ndarray]

    class Config:
        """Configuration for pydantic validation."""

        arbitrary_types_allowed = True

    @validator("trace")
    def validate_trace(cls, trace):
        """Validate a PyMC3 MultiTrace object.

        Args:
            trace ([type]): MultiTrace object.

        Raises:
            ValueError: If the object does not satisfy pre-determined requirements.

        Returns:
            [type]: The original object (if valid).
        """
        trace_methods = dir(trace)
        expected_methods = ["get_values"]
        for method in expected_methods:
            if method not in trace_methods:
                raise ValueError(
                    f"Object passed for trace does not have the method '{method}'."
                )
        return trace


class ApproximationSamplingResults(MCMCSamplingResults):
    """The results of ADVI fitting and sampling."""

    approximation: pm.Approximation


def convert_samples_to_arviz(
    model: pm.Model,
    res: Union[MCMCSamplingResults, ApproximationSamplingResults],
) -> az.InferenceData:
    """Turn the results from a sampling procedure into a standard ArviZ object.

    Args:
        model (pm.Model): The PyMC3 model.
        res (Union[MCMCSamplingResults, ApproximationSamplingResults]): The results of the sampling/fitting process.

    Returns:
        az.InferenceData: A standard ArviZ data object.
    """
    return az.from_pymc3(
        trace=res.trace,
        model=model,
        prior=res.prior_predictive,
        posterior_predictive=res.posterior_predictive,
    )


class SpecletModel:
    """Foundational model object in the Speclet project."""

    cache_dir: Optional[Path]

    def __init__(self, cache_dir: Optional[Path] = None):
        """Instantiate a new SpecletModel object.

        Args:
            cache_dir (Optional[Path], optional): The directory for caching sampling/fitting results. Defaults to None.
        """
        self.cache_dir = cache_dir

    def get_cache_file_names(self) -> ModelCachePaths:
        """Generate standard caching file paths.

        Raises:
            ValueError: Thrown if `self.cache_dir` is None.

        Returns:
            ModelCachePaths: Object containing the paths to use for caching various results from fitting a model.
        """
        if self.cache_dir is None:
            raise ValueError(
                "Cannot generate caching directory when `cache_dir` is None."
            )
        trace_dir_path = self.cache_dir / "pm-trace"
        prior_file_path = self.cache_dir / "prior-predictive-check.pkl"
        post_file_path = self.cache_dir / "posterior-predictive-check.pkl"
        approx_file_path = self.cache_dir / "vi-approximation.pkl"
        return ModelCachePaths(
            trace_path=trace_dir_path,
            prior_predictive_path=prior_file_path,
            posterior_predictive_path=post_file_path,
            approximation_path=approx_file_path,
        )

    def read_cached_sampling(self, model: pm.Model) -> MCMCSamplingResults:
        """Read sampling from cache.

        Args:
            model (pm.Model): The model corresponding to the cached sampling.

        Returns:
            MCMCSamplingResults: The cached data.
        """
        cache_paths = self.get_cache_file_names()

        trace = pm.load_trace(cache_paths.trace_path.as_posix(), model=model)
        post_check = self._get_pickle(cache_paths.posterior_predictive_path)
        prior_check = self._get_pickle(cache_paths.prior_predictive_path)

        return MCMCSamplingResults(
            trace=trace, prior_predictive=prior_check, posterior_predictive=post_check
        )

    def read_cached_approximation(
        self, draws: int = 1000
    ) -> ApproximationSamplingResults:
        """Read VI Approximation results from cache.

        Args:
            model (pm.Model): The model corresponding to the cached VI.

        Returns:
            ApproximationSamplingResults: The cached data.
        """
        cache_paths = self.get_cache_file_names()

        post_check = self._get_pickle(cache_paths.posterior_predictive_path)
        prior_check = self._get_pickle(cache_paths.prior_predictive_path)
        approx = self._get_pickle(cache_paths.approximation_path)
        trace = approx.sample(draws)

        return ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior_check,
            posterior_predictive=post_check,
            approximation=approx,
        )

    def cache_sampling_results(
        self, res: Union[MCMCSamplingResults, ApproximationSamplingResults]
    ) -> None:
        """Cache sampling results to disk.

        Args:
            res (Union[MCMCSamplingResults, ApproximationSamplingResults]): The results to cache.
        """
        cache_paths = self.get_cache_file_names()
        self._write_pickle(
            res.posterior_predictive, cache_paths.posterior_predictive_path
        )
        self._write_pickle(res.prior_predictive, cache_paths.prior_predictive_path)
        if isinstance(res, ApproximationSamplingResults):
            pm.save_trace(
                res.trace, directory=cache_paths.trace_path.as_posix(), overwrite=True
            )
            self._write_pickle(res.approximation, cache_paths.approximation_path)
        elif isinstance(res, MCMCSamplingResults):
            pm.save_trace(
                res.trace, directory=cache_paths.trace_path.as_posix(), overwrite=True
            )

    def _write_pickle(self, x: Any, fp: Path) -> None:
        with open(fp, "wb") as f:
            pickle.dump(x, f)
        return None

    def _get_pickle(self, fp: Path) -> Any:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        return d
