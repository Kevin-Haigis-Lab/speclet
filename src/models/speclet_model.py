#!/usr/bin/env python3

"""Foundational model object and related functions for the Speclet project."""

from pathlib import Path
from typing import Optional

import pymc3 as pm
from pydantic import BaseModel

from src.io import cache_io
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling.cache_manager import Pymc3CacheManager


class ModelCachePaths(BaseModel):
    """Paths for model caching."""

    trace_path: Path
    prior_predictive_path: Path
    posterior_predictive_path: Path
    approximation_path: Path


class SpecletModel:
    """Foundational model object in the Speclet project."""

    name: str
    cache_dir: Path
    mcmc_cache_delegate: Pymc3CacheManager
    advi_cache_delegate: Pymc3CacheManager

    def __init__(self, name: str, root_cache_dir: Optional[Path] = None):
        """Instantiate a new SpecletModel object.

        Args:
            name
            cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
        """
        self.name = name

        if root_cache_dir is None:
            root_cache_dir = cache_io.default_cache_dir()
        self.cache_dir = root_cache_dir / self.name
        self.mcmc_cache_delegate = Pymc3CacheManager(cache_dir=self.cache_dir / "mcmc")
        self.advi_cache_delegate = Pymc3CacheManager(cache_dir=self.cache_dir / "advi")

    def get_mcmc_cache(self, model: pm.Model) -> pmapi.MCMCSamplingResults:
        """Get MCMC sampling results from cache.

        Args:
            model (pm.Model): Corresponding PyMC3 model.

        Returns:
            pmapi.MCMCSamplingResults: MCMC sampling results.
        """
        return self.mcmc_cache_delegate.read_cached_sampling(model=model)

    def get_advi_cache(self, draws: int = 1000) -> pmapi.ApproximationSamplingResults:
        """Get ADVI fitting results from cache.

        Args:
            draws (int): Number of draws from the posterior to make for the trace.

        Returns:
            pmapi.ApproximationSamplingResults: ADVI fitting results.
        """
        return self.advi_cache_delegate.read_cached_approximation(draws=draws)

    def write_mcmc_cache(self, res: pmapi.MCMCSamplingResults) -> None:
        """Cache MCMC sampling results.

        Args:
            res (pmapi.MCMCSamplingResults): MCMC sampling results.
        """
        self.mcmc_cache_delegate.cache_sampling_results(res=res)

    def write_advi_cache(self, res: pmapi.ApproximationSamplingResults) -> None:
        """Cache ADVI fitting results.

        Args:
            res (pmapi.ApproximationSamplingResults): ADVI fitting results.
        """
        self.advi_cache_delegate.cache_sampling_results(res=res)

    def mcmc_cache_exists(self) -> bool:
        """Confirm that a cache of MCMC sampling results exists.

        Returns:
            bool: Does the cache exist?
        """
        return self.mcmc_cache_delegate.cache_exists(method="mcmc")

    def advi_cache_exists(self) -> bool:
        """Confirm that a cache of ADVI fitting results exists.

        Returns:
            bool: Does the cache exist?
        """
        return self.advi_cache_delegate.cache_exists(method="approx")
