"""Model posterior data manager."""

from pathlib import Path
from typing import Any, Callable

import arviz as az
import pandas as pd

from speclet import model_configuration as model_config
from speclet.bayesian_models import BayesianModelProtocol, get_bayesian_model
from speclet.io import DataFile, data_path, models_dir, project_root
from speclet.loggers import logger
from speclet.managers.cache_manager import PosteriorManager as PosteriorCacheManager
from speclet.managers.cache_manager import get_posterior_cache_name
from speclet.managers.data_managers import CrisprScreenDataManager, broad_only
from speclet.project_configuration import (
    get_model_configuration_file,
    project_config_broad_only,
)
from speclet.project_enums import ModelFitMethod


class PosteriorDataManager:
    """Posterior data manager."""

    def __init__(
        self,
        name: str,
        *,
        fit_method: ModelFitMethod,
        config_path: Path | None = None,
        posterior_dir: Path | None = None,
        broad_only: bool | None = None,
        id: str | None = None,
    ) -> None:
        """Posterior data manager.

        Args:
            name (str): Name of the model in the configuration file.
            fit_method (ModelFitMethod): Model fit method.
            config_path (Path | None, optional): Configuration file. Defaults to `None`
            in which case the model configuration file listed in the project
            configuration is used.
            posterior_dir (Path | None, optional): Directory with the posterior files.
            Defaults to `None` in which case the models directory in the project
            configuration is used.
            broad_only (bool | None, optional): Only use the Broad data. Defaults to
            `None` in which case the value in the project configuration is used.
            id (str | None, optional): Optional identifier instead of the name in the
            configuration file. Defaults to `None` (which is internally replaced by the
            model configuration name).
        """
        self.name = name
        self.fit_method = fit_method
        self.id = id if id is not None else name

        if config_path is None:
            config_path = project_root() / get_model_configuration_file()
        self._config_path = config_path
        self.config = model_config.get_configuration_for_model(
            config_path=config_path, name=name
        )

        if broad_only is None:
            broad_only = project_config_broad_only()
        self._broad_only = broad_only

        if posterior_dir is None:
            posterior_dir = models_dir()
        self._cache_name = get_posterior_cache_name(
            model_name=self.name, fit_method=fit_method
        )
        self.post_cache_manager = PosteriorCacheManager(
            id=self._cache_name, cache_dir=posterior_dir
        )

        # Properties to be acquired when needed.
        self._posterior_summary: pd.DataFrame | None = None
        self._trace: az.InferenceData | None = None
        self._bayes_model: BayesianModelProtocol | None = None
        self._data: pd.DataFrame | None = None
        self._valid_data: pd.DataFrame | None = None
        self._model_data_struct: Any | None = None
        return None

    @property
    def bayes_model(self) -> BayesianModelProtocol:
        """Load the Bayesian model object."""
        if self._bayes_model is None:
            self._bayes_model = get_bayesian_model(self.config.model)(
                **self.config.model_kwargs
            )
        return self._bayes_model

    @property
    def posterior_dir(self) -> Path:
        """Model's posterior directory."""
        return self.post_cache_manager.cache_path

    def read_description(self) -> str:
        """Read the description file."""
        desc_path = self.posterior_dir / "description.txt"
        with open(desc_path, "r") as file:
            return "".join(list(file))

    @property
    def posterior_summary(self) -> pd.DataFrame:
        """The summary of the model's posterior."""
        if self._posterior_summary is not None:
            return self._posterior_summary
        self._posterior_summary = pd.read_csv(
            self.posterior_dir / "posterior-summary.csv"
        ).assign(var_name=lambda d: [x.split("[")[0] for x in d["parameter"]])
        return self._posterior_summary

    @property
    def trace(self) -> az.InferenceData:
        """Model posterior trace object.

        If the model has a separate posterior predictive data file, the posterior is
        extended with this property.
        """
        if self._trace is not None:
            return self._trace

        _trace = self.post_cache_manager.get_posterior()
        assert _trace is not None, "Posterior data file not found."

        if self.post_cache_manager.posterior_predictive_cache_exists:
            _post_pred = self.post_cache_manager.get_posterior_predictive()
            assert _post_pred is not None
            _trace.extend(_post_pred, join="left")

        self._trace = _trace
        return self._trace

    @property
    def data(self) -> pd.DataFrame:
        """Data used to fit the data."""
        if self._data is None:
            trans = [broad_only] if self._broad_only else []
            self._data = CrisprScreenDataManager(
                self.data_file, transformations=trans
            ).get_data(read_kwargs={"low_memory": False})
        return self._data

    @property
    def data_file(self) -> Path:
        """Path to data file."""
        data_file = self.config.data_file
        if isinstance(data_file, Path):
            if data_file.exists():
                return data_file
            elif (project_root() / data_file).exists():
                return project_root() / data_file
            else:
                raise FileNotFoundError("Data file cannot be located.")
        elif isinstance(data_file, DataFile):
            return data_path(data_file)
        else:
            raise ValueError(f"Unexpected data file path type: {type(data_file)}")

    @property
    def valid_data(self) -> pd.DataFrame:
        """The data passed through the validation pipeline for the model."""
        if self._valid_data is not None:
            return self._valid_data

        validation_fxn: Callable[..., pd.DataFrame] | None = getattr(
            self.bayes_model, "data_processing_pipeline", None
        )
        if validation_fxn is None:
            raise NotImplementedError("No validation pipeline for model.")

        self._valid_data = validation_fxn(self.data.copy())
        return self._valid_data

    @property
    def model_data_struct(self) -> Any:
        """Model-specific data structure."""
        if self._model_data_struct is not None:
            return self._model_data_struct

        model_structure_fxn: Callable[[pd.DataFrame], Any] | None = getattr(
            self.bayes_model, "make_data_structure", None
        )
        if model_structure_fxn is None:
            raise NotImplementedError("No model structure generator for model.")

        self._model_data_struct = model_structure_fxn(self.valid_data)
        return self._model_data_struct

    def load_all(self) -> None:
        """Load all data objects now (instead of when first requested)."""
        _ = self.data
        try:
            _ = self.valid_data
            _ = self.model_data_struct
        except NotImplementedError:
            logger.debug("Data validation or model structure creation not available.")
        return None


class PosteriorDataManagers:
    """Collection of posterior data managers."""

    def __init__(
        self,
        names: list[str],
        *,
        fit_methods: list[ModelFitMethod] | ModelFitMethod,
        config_paths: list[Path | None] | Path | None = None,
        posterior_dirs: list[Path | None] | Path | None = None,
        broad_onlys: list[bool | None] | bool | None = None,
        keys: list[str] | None = None,
    ) -> None:
        """Collection of posterior data managers.

        Args:
            names (list[str]): NAmes from the model configuration(s).
            fit_methods (list[ModelFitMethod] | ModelFitMethod): Fit method(s) for the
            models.
            config_paths (list[Path | None] | Path | None, optional): Path(s) to
            configuration files. Defaults to None.
            posterior_dirs (list[Path | None] | Path | None, optional): Directory(ies)
            with the posterior files. Defaults to None.
            broad_onlys (list[bool | None] | bool | None, optional): Only use the Broad
            data (per model). Defaults to None.
            keys (list[str] | None, optional): Optional keys to use instead of the
            model's name in the configuration file. Defaults to None.
        """
        self.names = names.copy()
        self._n = len(self.names)

        # Typing stubs.
        self.fit_methods: list[ModelFitMethod]
        self.config_paths: list[Path | None]
        self.posterior_dirs: list[Path | None]
        self.broad_onlys: list[bool | None]
        self.keys: list[str]

        if isinstance(fit_methods, ModelFitMethod):
            self.fit_methods = [fit_methods] * self._n
        else:
            assert len(fit_methods) == self._n
            self.fit_methods = fit_methods.copy()

        if isinstance(config_paths, Path) or config_paths is None:
            self.config_paths = [config_paths] * self._n
        else:
            assert len(config_paths) == self._n
            self.config_paths = config_paths.copy()

        if isinstance(posterior_dirs, Path) or posterior_dirs is None:
            self.posterior_dirs = [posterior_dirs] * self._n
        else:
            assert len(posterior_dirs) == self._n
            self.posterior_dirs = posterior_dirs.copy()

        if isinstance(broad_onlys, bool) or broad_onlys is None:
            self.broad_onlys = [broad_onlys] * self._n
        else:
            assert len(broad_onlys) == self._n
            self.broad_onlys = broad_onlys.copy()

        if keys is None:
            self.keys = self.names.copy()
        else:
            assert len(keys) == self._n
            self.keys = keys.copy()

        self._posterior_data_managers: dict[str, PosteriorDataManager] = {}
        for i, key in enumerate(self.keys):
            pm = PosteriorDataManager(
                name=self.names[i],
                fit_method=self.fit_methods[i],
                config_path=self.config_paths[i],
                posterior_dir=self.posterior_dirs[i],
                broad_only=self.broad_onlys[i],
                id=key,
            )
            self._posterior_data_managers[key] = pm
        return None

    @property
    def posteriors(self) -> list[PosteriorDataManager]:
        """List all posterior data managers."""
        return list(self._posterior_data_managers.values())

    def as_dict(self) -> dict[str, PosteriorDataManager]:
        """Posterior data managers as a dictionary."""
        return self._posterior_data_managers.copy()

    def __getitem__(self, key: str) -> PosteriorDataManager:
        """Get a posterior manager."""
        return self._posterior_data_managers[key]

    def get(self, key: str, default: Any | None = None) -> PosteriorDataManager | None:
        """Get a posterior manager (with default is not available)."""
        return self._posterior_data_managers.get(key, default)

    def __len__(self) -> int:
        """Number of posterior managers."""
        return self._n
