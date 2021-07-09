#!/usr/bin/env python3

"""PyMC3 model cache manager."""

import pickle
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, Union

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


def _mkdir(dir: Path):
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


class Pymc3CacheManager:
    """PyMC3 model cache manager."""

    cache_dir: Path

    def __init__(self, cache_dir: Path):
        """Instantiate a new Pymc3CacheManager object.

        Args:
            cache_dir (Path): The directory for caching sampling/fitting results.
        """
        self.cache_dir = cache_dir
        _mkdir(self.cache_dir)

    def get_cache_file_names(self) -> ModelComponentCachePaths:
        """Generate standard caching file paths.

        Returns:
            ModelComponentCachePaths: Object containing the paths to use for caching
              various results from fitting a model.
        """
        trace_dir_path = self.cache_dir / "pm-trace"
        prior_file_path = self.cache_dir / "prior-predictive-check.pkl"
        post_file_path = self.cache_dir / "posterior-predictive-check.pkl"
        approx_file_path = self.cache_dir / "vi-approximation.pkl"
        return ModelComponentCachePaths(
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
        post_check = _get_pickle(cache_paths.posterior_predictive_path)
        prior_check = _get_pickle(cache_paths.prior_predictive_path)

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

        post_check = _get_pickle(cache_paths.posterior_predictive_path)
        prior_check = _get_pickle(cache_paths.prior_predictive_path)
        approx = _get_pickle(cache_paths.approximation_path)
        trace = approx.sample(draws)

        return pmapi.ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior_check,
            posterior_predictive=post_check,
            approximation=approx,
        )

    _SamplingResT = Union[pmapi.MCMCSamplingResults, pmapi.ApproximationSamplingResults]

    def cache_sampling_results(self, res: _SamplingResT) -> None:
        """Cache sampling results to disk.

        Args:
            res (_SamplingResT): The results to cache.
        """
        _mkdir(self.cache_dir)
        cache_paths = self.get_cache_file_names()
        _write_pickle(res.posterior_predictive, cache_paths.posterior_predictive_path)
        _write_pickle(res.prior_predictive, cache_paths.prior_predictive_path)
        if isinstance(res, pmapi.ApproximationSamplingResults):
            _write_pickle(res.approximation, cache_paths.approximation_path)
        elif isinstance(res, pmapi.MCMCSamplingResults):
            pm.save_trace(
                res.trace, directory=cache_paths.trace_path.as_posix(), overwrite=True
            )

    def cache_exists(self, method: ModelFitMethod) -> bool:
        """Confirm that the cached sampling/fitting results exist.

        This method checks for each pickle file and the trace directory (if applicable).

        Args:
            method (ModelFitMethod): Which cache to look for.

        Returns:
            bool: Does the cache exist?

        Raises:
            UnsupportedFitMethod: If the passed method is not an acceptable option.
        """
        cache_paths = self.get_cache_file_names()
        if (
            not cache_paths.prior_predictive_path.exists()
            or not cache_paths.posterior_predictive_path
        ):
            return False

        if method is ModelFitMethod.MCMC:
            return cache_paths.trace_path.exists()
        elif method is ModelFitMethod.ADVI:
            return cache_paths.approximation_path.exists()
        else:
            assert_never(method)

    def clear_cache(self) -> bool:
        """Remove the cache directory.

        Returns:
            bool: Whether the directory was removed or not. Returns False if the
              directory did not exist.
        """
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            return True
        return False


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
    ) -> Tuple[az.InferenceData, pm.Approximation]:
        """Read VI Approximation results from cache.

        Args:
            check_exists (bool, optional): Should the existence of the cache be checked
              first? Defaults to True.

        Raises:
            FileNotFoundError: If the cache does not exists (only if
            `check_exists = True`).

        Returns:
            Tuple[az.InferenceData, pm.Approximation]: The cached data.
        """
        logger.debug("Reading approximation cache from file.")
        cache_paths = self.get_cache_file_names()
        if check_exists:
            if not self.cache_exists(method=ModelFitMethod.ADVI):
                raise CacheDoesNotExistError(self.cache_dir)
        inf_data = self.read_cached_sampling(check_exists=False)
        approx = _get_pickle(cache_paths.approximation_path)
        return inf_data, approx

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
