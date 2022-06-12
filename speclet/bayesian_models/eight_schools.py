"""Classic eight-schools example."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm

from speclet.project_enums import ModelFitMethod


@dataclass(frozen=True)
class SchoolsData:
    """Data structure for the eight-schools data."""

    J: int
    y: list[float]
    sigma: list[int]


class EightSchoolsModel:
    """Classic eight-schools example model."""

    def __init__(self) -> None:
        """Classic eight-schools example model."""
        return None

    def vars_regex(self, fit_method: ModelFitMethod | None = None) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        return [r".*"]

    @property
    def schools_data(self) -> SchoolsData:
        """Eight-schools data."""
        return SchoolsData(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )

    def pymc_model(
        self,
        data: pd.DataFrame,
        skip_data_processing: bool = False,
    ) -> pm.Model:
        """PyMC3  model for a simple negative binomial model.

        Not implemented.

        Args:
            data (pd.DataFrame): Ignored.
            seed (Optional[seed], optional): Random seed. Defaults to `None`.
            skip_data_processing (bool, optional). Ignored.

        Returns:
            pm.Model: PyMC3 model.
        """
        school_data = self.schools_data
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfCauchy("tau", 5)
            theta_tilde = pm.Normal("eta", 0, 1)
            theta = pm.Deterministic("theta", mu + tau * theta_tilde)
            y = pm.Normal(  # noqa: F841
                name="y",
                mu=theta,
                sigma=np.array(school_data.sigma),
                observed=school_data.y,
            )
        return model
