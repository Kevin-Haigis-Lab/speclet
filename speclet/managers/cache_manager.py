"""Cache manager for Bayesian model posterior sampling data."""

from pathlib import Path
from typing import Optional, Union

import arviz as az

from speclet.project_enums import ModelFitMethod


class PosteriorManager:
    """Model posterior manager."""

    id: str
    cache_dir: Path
    _posterior: Optional[az.InferenceData]

    def __init__(self, id: str, cache_dir: Union[Path, str]) -> None:
        """Create a posterior manager.

        Args:
            id (str): Identifier of the posterior.
            cache_dir (Union[Path, str]): Directory for caching the posterior.
        """
        self.id = id
        self._posterior = None
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

    @property
    def cache_path(self) -> Path:
        """Path to the cache file."""
        return self.cache_dir / f"{self.id}_posterior.netcdf"

    @property
    def cache_exists(self) -> bool:
        """The cache file exists."""
        return self.cache_path.exists()

    def clear_cache(self) -> None:
        """Clear cached file."""
        if self.cache_exists:
            self.cache_path.unlink(missing_ok=False)
        return None

    def clear(self) -> None:
        """Clear posterior from file and in-memory store."""
        self._posterior = None
        self.clear_cache()
        return None

    def write_to_file(self) -> None:
        """If currently in memory, force the posterior object to be written to fil.."""
        if self._posterior is None:
            return None
        self._posterior.to_netcdf(str(self.cache_path))

    def put(self, trace: az.InferenceData) -> None:
        """Put a new posterior object to file.

        Args:
            trace (az.InferenceData): A model's posterior data.
        """
        self._posterior = trace
        trace.to_netcdf(str(self.cache_path))

    def get(self, from_file: bool = False) -> Optional[az.InferenceData]:
        """Get a model's posterior data.

        Args:
            from_file (bool, optional): Force re-reading the posterior from file.
            Defaults to False.

        Returns:
            Optional[az.InferenceData]: If it exists, the model's posterior data.
        """
        if not from_file and self._posterior is not None:
            return self._posterior
        elif self.cache_exists:
            return az.from_netcdf(str(self.cache_path))
        else:
            return None


def get_posterior_cache_name(model_name: str, fit_method: ModelFitMethod) -> str:
    """Create a posterior cache name using the model name and fit method.

    Args:
        model_name (str): Model name.
        fit_method (ModelFitMethod): Model fitting method used.

    Returns:
        str: Name for the posterior cache.
    """
    return model_name + "_" + fit_method.value.lower()


def cache_posterior(
    posterior: az.InferenceData, name: str, fit_method: ModelFitMethod, cache_dir: Path
) -> Path:
    """Cache the posterior of a model.

    Args:
        posterior (az.InferenceData): Posterior samples.
        name (str): Identifiable name of the model.
        fit_method (ModelFitMethod): Method used to fit the model.
        cache_dir (Path): Directory to write to.
    """
    id = get_posterior_cache_name(model_name=name, fit_method=fit_method)
    posterior_manager = PosteriorManager(id=id, cache_dir=cache_dir)
    posterior_manager.put(posterior)
    return posterior_manager.cache_path
