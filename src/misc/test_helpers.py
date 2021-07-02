"""Various functions used in many test modules."""

#### ---- src.models ---- ####

from random import choices
from typing import List, Optional, Tuple, Type, TypeVar

from src.models.speclet_five import SpecletFiveParameterization
from src.models.speclet_four import SpecletFourParameterization
from src.models.speclet_six import SpecletSixParameterization
from src.project_enums import ModelParameterization as MP

_ParameterizationT = TypeVar(
    "_ParameterizationT",
    SpecletFourParameterization,
    SpecletFiveParameterization,
    SpecletSixParameterization,
)

_param_options: List[MP] = [MP.CENTERED, MP.NONCENTERED]


def generate_model_parameterizations(
    param_class: Type[_ParameterizationT],
    n_randoms: int,
    param_options: Optional[List[MP]] = None,
) -> List[_ParameterizationT]:
    """Generate k random parameterization configurations.

    Args:
        param_class (Type[_ParameterizationT]): The parameterization class to use.
        n_randoms (int): Number of configurations to make. Since the process is random
          and duplicates are removed, there may be fewer than `k` configurations in
          the end.
        param_options (Optional[List[MP]], optional): Options to use for
          parameterization values. Defaults to `None` which uses a default list
          of `ModelParameterization`.

    Returns:
        List[_ParameterizationT]: A list of unique parameterization configurations.
    """
    if param_options is None:
        param_options = _param_options

    k = len(param_class._fields)
    mp_idx_tuples: List[Tuple[int, ...]] = [
        tuple(choices((0, 1), k=k)) for _ in range(n_randoms)
    ]
    mp_idx_tuples = list(set(mp_idx_tuples))

    model_parameterizations: List[_ParameterizationT] = [
        param_class(*[param_options[arg] for arg in args]) for args in mp_idx_tuples
    ]

    return model_parameterizations
