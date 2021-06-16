"""Models and enums for the pipelines."""


from enum import Enum

from pydantic import BaseModel

#### ---- Enums ---- ####


class ModelOption(str, Enum):
    """Model options."""

    speclet_test_model = "speclet-test-model"
    crc_ceres_mimic = "crc-ceres-mimic"
    speclet_one = "speclet-one"
    speclet_two = "speclet-two"
    speclet_three = "speclet-three"
    speclet_four = "speclet-four"
    speclet_five = "speclet-five"
    speclet_six = "speclet-six"
    speclet_seven = "speclet-seven"


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"


#### ---- Models ---- ####


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    model: ModelOption
    fit_method: ModelFitMethod
