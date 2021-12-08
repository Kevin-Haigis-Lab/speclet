"""Cache manager for Bayesian model posterior sampling data."""

from pathlib import Path
from typing import Optional, Union

import arviz as az


class PosteriorManager:

    id: str
    cache_dir: Path
    _posterior: Optional[az.InferenceData]

    def __init__(self, id: str, cache_dir: Union[Path, str]) -> None:
        self.id = id
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"{self.id})_posterior.netcdf"

    @property
    def cache_exists(self) -> bool:
        return self.cache_path.exists()

    def clear_cache(self) -> None:
        if self.cache_exists:
            self.cache_path.unlink(missing_ok=False)
        return None

    def write_to_file(self) -> None:
        if self._posterior is None:
            return None
        self._posterior.to_netcdf(str(self.cache_path))

    def put(self, trace: az.InferenceData) -> None:
        self._posterior = trace
        trace.to_netcdf(str(self.cache_path))

    def get(self, from_file: bool = False) -> Optional[az.InferenceData]:
        if not from_file and self._posterior is not None:
            return self._posterior
        elif self.cache_path.exists:
            return az.from_netcdf(str(self.cache_path))
        else:
            return None
