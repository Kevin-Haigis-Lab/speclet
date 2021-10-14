"""Available cache managers for different data types."""

import pickle
import shutil
from pathlib import Path
from typing import Any, Optional

import arviz as az
import pymc3 as pm
from pydantic import BaseModel

from src.exceptions import CacheDoesNotExistError
from src.loggers import logger
from src.modeling import pymc3_sampling_api as pmapi
from src.project_enums import ModelFitMethod, assert_never


class ModelComponentCachePaths(BaseModel):
    """Paths for caching model components."""

    trace_path: Path
    prior_predictive_path: Path
    posterior_predictive_path: Path
    approximation_path: Path


class ArvizCachePaths(BaseModel):
    """Paths for caching ArviZ data."""

    inference_data_path: Path
    approximation_path: Path


def _mkdir(dir: Path) -> None:
    if not dir.exists():
        dir.mkdir(parents=True)


def _write_pickle(x: Any, fp: Path) -> None:
    with open(fp, "wb") as f:
        pickle.dump(x, f)
    return None


def _get_pickle(fp: Path) -> Any:
    with open(fp, "rb") as f:
        d = pickle.load(f)
    return d


class ArvizCacheManager:
    """Cache of model results using ArviZ's InferenceData object."""

    cache_dir: Path

    def __init__(self, cache_dir: Path):
        """Instantiate a new ArvizCacheManager object.

        Args:
            cache_dir (Path): The directory for caching sampling/fitting results.
        """
        self.cache_dir = cache_dir
        _mkdir(self.cache_dir)

    def get_cache_file_names(self) -> ArvizCachePaths:
        """Generate standard caching file paths.

        Returns:
            ArvizCachePaths: Object containing the paths to use for caching the ArviZ
              InferenceObject and approximation object.
        """
        inference_data_file_path = self.cache_dir / "inference-data.nc"
        approx_file_path = self.cache_dir / "vi-approximation.pkl"
        return ArvizCachePaths(
            inference_data_path=inference_data_file_path,
            approximation_path=approx_file_path,
        )

    def cache_sampling_results(
        self,
        inference_data: az.InferenceData,
        approximation: Optional[pm.Approximation] = None,
    ) -> None:
        """Cache sampling results to disk.

        Args:
            inference_data (az.InferenceData): ArviZ InferenceData with sampling
              results.
            approximation (Optional[pm.Approximation], optional): Approximation results
              for VI methods. Defaults to None.
        """
        _mkdir(self.cache_dir)
        cache_paths = self.get_cache_file_names()
        logger.info(
            f"Caching InferenceData to '{cache_paths.inference_data_path.as_posix()}'."
        )
        inference_data.to_netcdf(filename=cache_paths.inference_data_path.as_posix())
        if approximation is not None:
            logger.info(
                f"Caching approx. to '{cache_paths.approximation_path.as_posix()}'."
            )
            _write_pickle(approximation, cache_paths.approximation_path)

    def cache_exists(self, method: ModelFitMethod) -> bool:
        """Confirm that the cached sampling/fitting results exist.

        Args:
            method (ModelFitMethod): Which cache to look for.

        Returns:
            bool: Does the cache exist?
        """
        cache_paths = self.get_cache_file_names()
        if not cache_paths.inference_data_path.exists():
            return False

        if method is ModelFitMethod.MCMC:
            # Nothing special to add.
            logger.info("ArvizCacheManager: MCMC cache exists.")
        elif method is ModelFitMethod.ADVI:
            if not cache_paths.approximation_path.exists():
                return False
            logger.info("ArvizCacheManager: ADVI cache exists.")
        else:
            assert_never(method)

        return True

    def read_cached_sampling(self, check_exists: bool = True) -> az.InferenceData:
        """Read sampling from cache.

        Args:
            check_exists (bool, optional): Should the existence of the cache be checked
              first? Defaults to True.

        Raises:
            FileNotFoundError: If the cache does not exists (only if
            `check_exists = True`).

        Returns:
            az.InferenceData: The cached data.
        """
        logger.debug("Reading sampling cache from file.")
        cache_paths = self.get_cache_file_names()
        if check_exists:
            if not self.cache_exists(method=ModelFitMethod.MCMC):
                raise CacheDoesNotExistError(self.cache_dir)
        return az.from_netcdf(cache_paths.inference_data_path.as_posix())

    def read_cached_approximation(
        self, check_exists: bool = True
    ) -> pmapi.ApproximationSamplingResults:
        """Read VI Approximation results from cache.

        Args:
            check_exists (bool, optional): Should the existence of the cache be checked
              first? Defaults to True.

        Raises:
            FileNotFoundError: If the cache does not exists (only if
            `check_exists = True`).

        Returns:
            tuple[az.InferenceData, pm.Approximation]: The cached data.
        """
        logger.debug("Reading approximation cache from file.")
        cache_paths = self.get_cache_file_names()
        if check_exists:
            if not self.cache_exists(method=ModelFitMethod.ADVI):
                raise CacheDoesNotExistError(self.cache_dir)
        inf_data = self.read_cached_sampling(check_exists=False)
        approx = _get_pickle(cache_paths.approximation_path)
        return pmapi.ApproximationSamplingResults(
            inference_data=inf_data, approximation=approx
        )

    def clear_cache(self) -> bool:
        """Remove the cache directory.

        Returns:
            bool: Whether the directory was removed or not. Returns False if the
              directory did not exist.
        """
        logger.debug("Trying to clear cache.")
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            return True
        return False
