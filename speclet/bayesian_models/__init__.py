"""Organization of the Bayesian model classes."""

from enum import Enum, unique
from typing import Any, Final, Optional, Protocol, Type

import pandas as pd
import pymc3 as pm
from stan.model import Model as StanModel

from speclet.bayesian_models.eight_schools import EightSchoolsModel
from speclet.bayesian_models.hierarchical_nb import HierarchcalNegativeBinomialModel
from speclet.bayesian_models.negative_binomial import NegativeBinomialModel
from speclet.project_enums import ModelFitMethod


@unique
class BayesianModel(Enum):
    """Available Bayesian models."""

    SIMPLE_NEGATIVE_BINOMIAL = "SIMPLE_NEGATIVE_BINOMIAL"
    EIGHT_SCHOOLS = "EIGHT_SCHOOLS"
    HIERARCHICAL_NB = "HIERARCHICAL_NB"


class BayesianModelProtocol(Protocol):
    """Protocol for Bayesian model objects."""

    def __init__(self) -> None:
        """Simple initialization method."""
        ...

    def vars_regex(self, fit_method: ModelFitMethod) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        ...

    def stan_model(
        self, data: pd.DataFrame, random_seed: Optional[int] = None
    ) -> StanModel:
        """Make a Stan model.

        Args:
            data (pd.DataFrame): Data to model.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            StanModel: A Stan model.
        """
        ...

    def stan_idata_addons(self, data: pd.DataFrame) -> dict[str, Any]:
        """Information to add to the InferenceData posterior object."""
        ...

    def pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """Make a PyMC3 model.

        Args:
            data (pd.DataFrame): Data to model.

        Returns:
            pm.Model: A PyMC3 model.
        """
        ...


BAYESIAN_MODEL_LOOKUP: Final[dict[BayesianModel, Type[BayesianModelProtocol]]] = {
    BayesianModel.EIGHT_SCHOOLS: EightSchoolsModel,
    BayesianModel.SIMPLE_NEGATIVE_BINOMIAL: NegativeBinomialModel,
    BayesianModel.HIERARCHICAL_NB: HierarchcalNegativeBinomialModel,
}


def get_bayesian_model(mdl: BayesianModel) -> Type[BayesianModelProtocol]:
    """Get a Bayesian model object.

    Args:
        mdl (BayesianModel): Bayesian model.

    Raises:
        BaseException: If there is no corresponding Bayesian model.

    Returns:
        BayesianModelProtocol: Object that implements the descired Bayesian model.
    """
    if mdl not in BAYESIAN_MODEL_LOOKUP:
        raise BaseException("Could not find the class for a Bayesian model.")
    return BAYESIAN_MODEL_LOOKUP[mdl]
