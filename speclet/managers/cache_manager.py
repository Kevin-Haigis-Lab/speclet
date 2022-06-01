"""Cache manager for Bayesian model posterior sampling data."""

from pathlib import Path
from typing import Optional

import arviz as az

from speclet.project_enums import ModelFitMethod


class PosteriorManager:
    """Model posterior manager."""

    def __init__(self, id: str, cache_dir: Path | str) -> None:
        """Create a posterior manager.

        The cache directory should be the general location for caching model results. A
        directory will be made for this specific model using its ID. Then, all cached
        files will live within this directory under standardized names.

        Args:
            id (str): Identifier of the posterior.
            cache_dir (Path | str): Directory for caching the posterior.
        """
        self.id = id
        self._posterior = None
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

    @property
    def cache_path(self) -> Path:
        """Path to the cache."""
        return self.cache_dir / self.id

    @property
    def posterior_path(self) -> Path:
        """Path to the cache file."""
        return self.cache_path / "posterior.netcdf"

    @property
    def posterior_cache_exists(self) -> bool:
        """The cache file exists."""
        return self.posterior_path.exists()

    def _make_dir(self) -> None:
        """Make the cache directory for this posterior."""
        if self.cache_path.exists():
            return
        self.cache_path.mkdir(parents=True)

    def clear_cache(self) -> None:
        """Clear cached file."""
        if self.posterior_cache_exists:
            self.posterior_path.unlink(missing_ok=False)
        return None

    def clear(self) -> None:
        """Clear posterior from file and in-memory store."""
        self._posterior = None
        self.clear_cache()
        return None

    def write_to_file(self) -> None:
        """If currently in memory, force the posterior object to be written to file."""
        self._make_dir()
        if self._posterior is None:
            return None
        self._posterior.to_netcdf(str(self.posterior_path))

    def put(self, trace: az.InferenceData) -> None:
        """Put a new posterior object to file.

        Args:
            trace (az.InferenceData): A model's posterior data.
        """
        self._make_dir()
        self._posterior = trace
        self.write_to_file()

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
        elif self.posterior_cache_exists:
            return az.from_netcdf(str(self.posterior_path))
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
    return model_name + "_" + fit_method.value


def cache_posterior(posterior: az.InferenceData, id: str, cache_dir: Path) -> Path:
    """Cache the posterior of a model.

    Args:
        posterior (az.InferenceData): Posterior samples.
        id (str): Uniquely identifiable ID passed to the PosteriorManager.
        cache_dir (Path): Directory to write to.
    """
    posterior_manager = PosteriorManager(id=id, cache_dir=cache_dir)
    posterior_manager.put(posterior)
    return posterior_manager.posterior_path


def get_cached_posterior(id: str, cache_dir: Path) -> az.InferenceData:
    """Retrieve a posterior from cache.

    Args:
        id (str): Uniquely identifiable ID passed to the PosteriorManager.
        cache_dir (Path): Directory containing the cache.

    Raises:
        FileNotFoundError: Error thrown if the posterior cache does not exist.

    Returns:
        az.InferenceData: Posterior trace.
    """
    posterior_manager = PosteriorManager(id=id, cache_dir=cache_dir)
    trace = posterior_manager.get()
    if trace is None:
        raise FileNotFoundError(posterior_manager.posterior_path)
    return trace
