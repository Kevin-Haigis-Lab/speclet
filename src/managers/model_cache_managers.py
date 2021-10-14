"""Managers for caching and retrieval of model fitting results."""

from pathlib import Path
from typing import Optional

import arviz as az
from pydantic import BaseModel

from src.io import cache_io
from src.managers.cache_managers import ArvizCacheManager
from src.modeling.pymc3_sampling_api import ApproximationSamplingResults
from src.project_enums import ModelFitMethod


class ModelCachePaths(BaseModel):
    """Paths for model caching."""

    trace_path: Path
    prior_predictive_path: Path
    posterior_predictive_path: Path
    approximation_path: Path


class Pymc3ModelCacheManager:
    """Object for managing the caches of the results of fitting PyMC3 models."""

    name: str
    cache_dir: Path
    mcmc_cache_delegate: ArvizCacheManager
    advi_cache_delegate: ArvizCacheManager

    def __init__(self, name: str, root_cache_dir: Optional[Path] = None):
        """Instantiate a new SpecletModel object.

        Args:
            name (str): Identifiable name of the model to be incorporated in the cache
              directory name.
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None to use the project default
              cache directory.
        """
        self.name = name

        if root_cache_dir is None:
            root_cache_dir = cache_io.default_cache_dir()
        self.cache_dir = root_cache_dir / self.name
        self.mcmc_cache_delegate = ArvizCacheManager(cache_dir=self.cache_dir / "mcmc")
        self.advi_cache_delegate = ArvizCacheManager(cache_dir=self.cache_dir / "advi")

    def get_mcmc_cache(self) -> az.InferenceData:
        """Get MCMC sampling results from cache.

        Args:
            model (pm.Model): Corresponding PyMC3 model.

        Returns:
            pmapi.MCMCSamplingResults: MCMC sampling results.
        """
        return self.mcmc_cache_delegate.read_cached_sampling()

    def get_advi_cache(self) -> ApproximationSamplingResults:
        """Get ADVI fitting results from cache.

        Args:
            draws (int): Number of draws from the posterior to make for the trace.

        Returns:
            ApproximationSamplingResults: ADVI fitting results.
        """
        return self.advi_cache_delegate.read_cached_approximation()

    def write_mcmc_cache(self, inference_data: az.InferenceData) -> None:
        """Cache MCMC sampling results.

        Args:
            inference_data (az.InferenceData): MCMC sampling results.
        """
        self.mcmc_cache_delegate.cache_sampling_results(inference_data)

    def write_advi_cache(
        self, approx_sampling_results: ApproximationSamplingResults
    ) -> None:
        """Cache ADVI fitting results.

        Args:
            approx_sampling_results (ApproximationSamplingResults): ADVI sampling
              results.
        """
        self.advi_cache_delegate.cache_sampling_results(
            approx_sampling_results.inference_data,
            approx_sampling_results.approximation,
        )

    def mcmc_cache_exists(self) -> bool:
        """Confirm that a cache of MCMC sampling results exists.

        Returns:
            bool: Does the cache exist?
        """
        return self.mcmc_cache_delegate.cache_exists(method=ModelFitMethod.MCMC)

    def advi_cache_exists(self) -> bool:
        """Confirm that a cache of ADVI fitting results exists.

        Returns:
            bool: Does the cache exist?
        """
        return self.advi_cache_delegate.cache_exists(method=ModelFitMethod.ADVI)

    def clear_mcmc_cache(self) -> None:
        """Clear the MCMC cache."""
        self.mcmc_cache_delegate.clear_cache()

    def clear_advi_cache(self) -> None:
        """Clear the ADVI cache."""
        self.advi_cache_delegate.clear_cache()

    def clear_all_caches(self) -> None:
        """Clear both the MCMC and ADVI cache."""
        self.clear_mcmc_cache()
        self.clear_advi_cache()
