"""Speclet Simple."""

from pathlib import Path
from typing import Optional

import pymc3 as pm
import theano

from src.io.data_io import DataFile

# from src.managers.model_data_managers import CrcDataManager, DataManager
from src.managers.data_managers import CrisprScreenDataManager
from src.models.speclet_model import (
    ObservedVarName,
    ReplacementsDict,
    SpecletModel,
    SpecletModelDataManager,
)


class SpecletSimple(SpecletModel):
    """SpecletSimple Model.

    $$
    \\begin{aligned}
    lfc &\\sim N(a, \\sigma) \\\\
    a &\\sim N(0, 5) \\\\
    \\sigma &\\sim \\text{Halfnormal}(0, 5) \\\\
    \\end{aligned}
    $$

    This is just a simple model for helping with testing pipelines and analyses.
    """

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[SpecletModelDataManager] = None,
    ):
        """Instantiate a SpecletSimple model.

        Args:
            name (str): A unique identifier for this instance of SpecletSimple. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is created
              automatically.
        """
        if data_manager is None:
            data_manager = CrisprScreenDataManager(DataFile.DEPMAP_CRC_SUBSAMPLE)

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Build SpecletSimple model.

        Returns:
            tuple[pm.Model, ObservedVarName]: The model and name of the observed
            variable.
        """
        data = self.data_manager.get_data()
        total_size = data.shape[0]
        lfc_shared = theano.shared(data.lfc.values)
        self.shared_vars = {
            "lfc_shared": lfc_shared,
        }

        with pm.Model() as model:
            # sgRNA|gene varying intercept.
            a = pm.Normal("a", 0, 5)
            σ = pm.HalfNormal("σ", 5)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc", a, σ, observed=lfc_shared, total_size=total_size
            )

        return model, "lfc"

    def get_replacement_parameters(self) -> ReplacementsDict:
        """Make a dictionary mapping the shared data variables to new data.

        Raises:
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        data = self.data_manager.get_data()
        batch_size = self._get_batch_size()
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }
