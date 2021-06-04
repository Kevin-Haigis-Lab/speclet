"""A simple, light-weight SpecletModel for testing pipelines."""

from pathlib import Path
from typing import Optional, Tuple

import pymc3 as pm

from src.managers.model_data_managers import DataManager, MockDataManager
from src.models.speclet_model import SpecletModel


class SpecletTestModel(SpecletModel):
    """A SpecletModel class for testing purposes."""

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
    ):
        """Initialize a SpecletTestModel.

        Args:
            name (str): Unique name of the model. 'TestingModel-' will be prepended.
            root_cache_dir (Optional[Path], optional): Location for cache. Defaults to
              None.
            debug (bool, optional): In debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `MockDataManager` is created automatically.
        """
        if data_manager is None:
            data_manager = MockDataManager(debug=debug)

        super().__init__(
            name="TestingModel-" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Specify a simple model.

        Returns:
            Tuple[pm.Model, str]: Model and name of target  variable.
        """
        d = self.data_manager.get_data()
        with pm.Model() as model:
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 5)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y", a + b * d.x.values, sigma, observed=d.y.values
            )
        return model, "y"

    def update_mcmc_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        self.mcmc_sampling_params.chains = 2
        self.mcmc_sampling_params.draws = 1000
        self.mcmc_sampling_params.tune = 1000
        self.mcmc_sampling_params.target_accept = 0.85
        return None

    def update_advi_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        self.advi_sampling_params.n_iterations = 10000
        return None
