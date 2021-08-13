"""Common types used throughout the project."""


from typing import TypeVar, Union

from src.models.ceres_mimic import CeresMimic, CeresMimicConfiguration
from src.models.speclet_five import SpecletFive, SpecletFiveConfiguration
from src.models.speclet_four import SpecletFour, SpecletFourConfiguration
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.models.speclet_seven import SpecletSeven, SpecletSevenConfiguration
from src.models.speclet_simple import SpecletSimple
from src.models.speclet_six import SpecletSix, SpecletSixConfiguration
from src.models.speclet_two import SpecletTwo

SpecletProjectModelTypes = Union[
    SpecletTestModel,
    SpecletSimple,
    CeresMimic,
    SpecletOne,
    SpecletTwo,
    SpecletFour,
    SpecletFive,
    SpecletSix,
    SpecletSeven,
]

ModelConfigurationT = TypeVar(
    "ModelConfigurationT",
    CeresMimicConfiguration,
    SpecletFiveConfiguration,
    SpecletFourConfiguration,
    SpecletSixConfiguration,
    SpecletSevenConfiguration,
)
