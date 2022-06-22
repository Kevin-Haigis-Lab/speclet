"""Organization of the Bayesian model classes."""

from enum import Enum, unique
from typing import Any, Final, Protocol

import pandas as pd
import pymc as pm

from speclet.bayesian_models.eight_schools import EightSchoolsModel
from speclet.bayesian_models.hierarchical_nb import HierarchicalNegativeBinomialModel
from speclet.bayesian_models.hierarchical_nb_secondtier import (
    HierarchcalNegativeBinomialSecondTier,
)
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
from speclet.bayesian_models.negative_binomial import NegativeBinomialModel
from speclet.project_enums import ModelFitMethod


@unique
class BayesianModel(Enum):
    """Available Bayesian models."""

    SIMPLE_NEGATIVE_BINOMIAL = "SIMPLE_NEGATIVE_BINOMIAL"
    EIGHT_SCHOOLS = "EIGHT_SCHOOLS"
    HIERARCHICAL_NB = "HIERARCHICAL_NB"
    HIERARCHICAL_NB_SECONDTIER = "HIERARCHICAL_NB_SECONDTIER"
    LINEAGE_HIERARCHICAL_NB = "LINEAGE_HIERARCHICAL_NB"


class BayesianModelProtocol(Protocol):
    """Protocol for Bayesian model objects."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with keyword-only arguments.

        Best practice is to pass this datainto a Pydantic configuration model.
        """
        ...

    def vars_regex(self, fit_method: ModelFitMethod | None = None) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        ...

    def pymc_model(
        self,
        data: pd.DataFrame,
        skip_data_processing: bool = False,
    ) -> pm.Model:
        """Make a PyMC model.

        Args:
            data (pd.DataFrame): Data to model.
            skip_data_processing (bool, optional). Skip data pre-processing step?
            Defaults to `False`.

        Returns:
            pm.Model: A PyMC model.
        """
        ...


BAYESIAN_MODEL_LOOKUP: Final[dict[BayesianModel, type[BayesianModelProtocol]]] = {
    BayesianModel.EIGHT_SCHOOLS: EightSchoolsModel,
    BayesianModel.SIMPLE_NEGATIVE_BINOMIAL: NegativeBinomialModel,
    BayesianModel.HIERARCHICAL_NB: HierarchicalNegativeBinomialModel,
    BayesianModel.HIERARCHICAL_NB_SECONDTIER: HierarchcalNegativeBinomialSecondTier,
    BayesianModel.LINEAGE_HIERARCHICAL_NB: LineageHierNegBinomModel,
}


def get_bayesian_model(mdl: BayesianModel) -> type[BayesianModelProtocol]:
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
