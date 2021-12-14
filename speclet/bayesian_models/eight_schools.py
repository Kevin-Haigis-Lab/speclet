"""Classic eight-schools example."""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pymc3 as pm
import stan
from stan.model import Model as StanModel

_eight_schools_stan_code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""


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

    @property
    def schools_data(self) -> SchoolsData:
        """Eight-schools data."""
        return SchoolsData(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )

    def stan_model(
        self, data: pd.DataFrame, random_seed: Optional[int] = None
    ) -> StanModel:
        """Stan model for a the eight-schools example.

        Args:
            data (pd.DataFrame): Ignored.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            StanModel: Stan model.
        """
        return stan.build(
            _eight_schools_stan_code,
            data=asdict(self.schools_data),
            random_seed=random_seed,
        )

    def pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """PyMC3  model for a simple negative binomial model.

        Not implemented.

        Args:
            data (pd.DataFrame): Ignored.

        Returns:
            pm.Model: PyMC3 model.
        """
        school_data = self.schools_data
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfNormal("tau", 5)
            eta = pm.Normal("eta", 0, 1)
            theta = pm.Deterministic("theta", mu + tau * eta)
            y = pm.Normal(  # noqa: F841
                name="y",
                mu=theta,
                sigma=np.array(school_data.sigma),
                observed=school_data.y,
            )
        return model
