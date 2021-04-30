#!/usr/bin/env python3

"""PyMC3 model cache manager."""

import os
import pickle
from pathlib import Path
from typing import Any, Union

import pymc3 as pm
from pydantic import BaseModel

from src.modeling import pymc3_sampling_api as pmapi


class ModelCachePaths(BaseModel):
    """Paths for model caching."""

    trace_path: Path
    prior_predictive_path: Path
    posterior_predictive_path: Path
    approximation_path: Path


class Pymc3CacheManager:
    """PyMC3 model cache manager."""

    cache_dir: Path

    def __init__(self, cache_dir: Path):
        """Instantiate a new SpecletModel object.

        Args:
            cache_dir (Path): The directory for caching sampling/fitting results.
              Defaults to None.
        """
        self.cache_dir = cache_dir
        self._mkdir()

    def _mkdir(self):
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def get_cache_file_names(self) -> ModelCachePaths:
        """Generate standard caching file paths.

        Raises:
            ValueError: Thrown if `self.cache_dir` is None.

        Returns:
            ModelCachePaths: Object containing the paths to use for caching
              various results from fitting a model.
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

    def read_cached_sampling(self, model: pm.Model) -> pmapi.MCMCSamplingResults:
        """Read sampling from cache.

        Args:
            model (pm.Model): The model corresponding to the cached sampling.

        Returns:
            pmapi.MCMCSamplingResults: The cached data.
        """
        cache_paths = self.get_cache_file_names()

        trace = pm.load_trace(cache_paths.trace_path.as_posix(), model=model)
        post_check = self._get_pickle(cache_paths.posterior_predictive_path)
        prior_check = self._get_pickle(cache_paths.prior_predictive_path)

        return pmapi.MCMCSamplingResults(
            trace=trace, prior_predictive=prior_check, posterior_predictive=post_check
        )

    def read_cached_approximation(
        self, draws: int = 1000
    ) -> pmapi.ApproximationSamplingResults:
        """Read VI Approximation results from cache.

        Args:
            draws (int, optional): The number of draws from the trace. Default is 1000.

        Returns:
            pmapi.ApproximationSamplingResults: The cached data.
        """
        cache_paths = self.get_cache_file_names()

        post_check = self._get_pickle(cache_paths.posterior_predictive_path)
        prior_check = self._get_pickle(cache_paths.prior_predictive_path)
        approx = self._get_pickle(cache_paths.approximation_path)
        trace = approx.sample(draws)

        return pmapi.ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior_check,
            posterior_predictive=post_check,
            approximation=approx,
        )

    def cache_sampling_results(
        self, res: Union[pmapi.MCMCSamplingResults, pmapi.ApproximationSamplingResults]
    ) -> None:
        """Cache sampling results to disk.

        Args:
            res (Union[pmapi.MCMCSamplingResults, pmapi.ApproximationSamplingResults]):
              The results to cache.
        """
        self._mkdir()
        cache_paths = self.get_cache_file_names()
        self._write_pickle(
            res.posterior_predictive, cache_paths.posterior_predictive_path
        )
        self._write_pickle(res.prior_predictive, cache_paths.prior_predictive_path)
        if isinstance(res, pmapi.ApproximationSamplingResults):
            self._write_pickle(res.approximation, cache_paths.approximation_path)
        elif isinstance(res, pmapi.MCMCSamplingResults):
            pm.save_trace(
                res.trace, directory=cache_paths.trace_path.as_posix(), overwrite=True
            )

    def cache_exists(self, method: str) -> bool:
        """Confirm that the cached sampling/fitting results exist.

        This method checks for each pickle file and the trace directory (if applicable).

        Args:
            method (str): Which cache to look for (either "mcmc" or "approx").

        Returns:
            bool: Does the cache exist?

        Raises:
            ValueError: If the passed method is not an acceptable option.
        """
        cache_paths = self.get_cache_file_names()
        if (
            not cache_paths.prior_predictive_path.exists()
            or not cache_paths.posterior_predictive_path
        ):
            return False

        if method == "mcmc":
            return cache_paths.trace_path.exists()
        elif method == "approx":
            return cache_paths.approximation_path.exists()
        else:
            raise ValueError(f"Unknown method '{method}'.")

    def _write_pickle(self, x: Any, fp: Path) -> None:
        with open(fp, "wb") as f:
            pickle.dump(x, f)
        return None

    def _get_pickle(self, fp: Path) -> Any:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        return d

    def clear_cache(self) -> bool:
        """Remove the cache directory.

        Returns:
            bool: Whether the directory was removed or not. Returns False if the
              directory did not exist.
        """
        if self.cache_dir.exists():
            os.rmdir(self.cache_dir)
            return True
        return False
