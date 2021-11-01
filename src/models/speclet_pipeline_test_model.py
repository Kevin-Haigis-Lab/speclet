"""A simple, light-weight SpecletModel for testing pipelines."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pymc3 as pm
from pydantic import BaseModel

from src.io.data_io import DataFile
from src.managers.data_managers import CrisprScreenDataManager
from src.models.speclet_model import (
    ObservedVarName,
    SpecletModel,
    SpecletModelDataManager,
)
from src.project_enums import ModelParameterization as MP


class SpecletTestModelConfiguration(BaseModel):
    """Parameterizations for each covariate in SpecletTestModel model."""

    some_param: MP = MP.CENTERED
    another_param: MP = MP.NONCENTERED
    cov1: bool = True
    cov2: bool = False


class SpecletTestModel(SpecletModel):
    """A SpecletModel class for testing purposes."""

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        data_manager: Optional[SpecletModelDataManager] = None,
        config: Optional[SpecletTestModelConfiguration] = None,
    ):
        """Initialize a SpecletTestModel.

        Args:
            name (str): Unique name of the model. 'TestingModel-' will be prepended.
            root_cache_dir (Optional[Path], optional): Location for cache. Defaults to
              None.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (Optional[SpecletTestModelConfiguration], optional): Model
              configuration.
        """
        if data_manager is None:
            data_manager = CrisprScreenDataManager(DataFile.DEPMAP_CRC_SUBSAMPLE)

        if config is None:
            self.config = SpecletTestModelConfiguration()
        else:
            self.config = config

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            data_manager=data_manager,
        )

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        self.config = SpecletTestModelConfiguration(**info)

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Specify a simple model.

        Returns:
            Tuple[pm.Model, ObservedVarName]: Model and name of target  variable.
        """
        d = self.data_manager.get_data()
        with pm.Model() as model:
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 5)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y", a + b * d.lfc.values, sigma, observed=d.lfc.values
            )
        return model, "y"

    def update_observed_data(self, new_data: np.ndarray) -> None:
        """Do nothing for this testing model."""
        return None
