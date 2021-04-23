"""Base class to contain a PyMC3 model."""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional

import pymc3 as pm
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.managers.model_data_managers import DataManager
from src.modeling import pymc3_sampling_api as pmapi


class SpecletModel:
    """Base class to contain a PyMC3 model."""

    name: str
    debug: bool
    cache_manager: Pymc3ModelCacheManager
    data_manager: Optional[DataManager] = None

    model: Optional[pm.Model] = None
    shared_vars: Optional[Dict[str, TTShared]] = None
    advi_results: Optional[pmapi.ApproximationSamplingResults] = None
    mcmc_results: Optional[pmapi.MCMCSamplingResults] = None

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
    ) -> None:
        """Instantiate a Speclet Model.

        Args:
            name (str): Name of the model.
            root_cache_dir (Optional[Path], optional): Location for the cache directory.
              If None (default), then the project's default cache directory is used.
              Defaults to None.
            debug (bool, optional): Use debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. Defaults to None.
        """
        self.name = name
        self.debug = debug
        self.cache_manager = Pymc3ModelCacheManager(
            name=name, root_cache_dir=root_cache_dir
        )
        if data_manager is not None:
            self.data_manager = data_manager
            self.data_manager.debug = self.debug

    @abstractmethod
    def model_specification(self) -> pm.Model:
        """Define the PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
        """
        pass

    def build_model(self) -> None:
        """Build the PyMC3 model.

        Raises:
            AttributeError: Raised if there is no data manager.
            AttributeError: Raised the `model` attribute is still None after calling
              `self.model_specification()`
        """
        if self.data_manager is None:
            raise AttributeError("Cannot build a model without a data manager.")

        self.model = self.model_specification()

        if self.model is None:
            m = "The `model` attribute cannot be None at the end of the "
            m += "`build_model()` method."
            raise AttributeError(m)

        return None

    # def mcmc_sample_model(self) -> pmapi.MCMCSamplingResults:
    #     # TODO
    #     return None

    # def advi_sample_model(self) -> pmapi.ApproximationSamplingResults:
    #     # TODO
    #     return None

    # def run_simulation_based_calibration(
    #     self, results_path: Path, random_seed: Optional[int] = None,
    # size: str = "large"
    # ) -> None:
    #     # TODO
    #     return None
