"""Models and enums for the pipelines."""


from enum import Enum, unique

from pydantic import BaseModel

from src.project_enums import ModelFitMethod

#### ---- Enums ---- ####


@unique
class ModelOption(str, Enum):
    """Model options."""

    speclet_test_model = "speclet-test-model"
    crc_ceres_mimic = "crc-ceres-mimic"
    speclet_one = "speclet-one"
    speclet_two = "speclet-two"
    speclet_four = "speclet-four"
    speclet_five = "speclet-five"
    speclet_six = "speclet-six"
    speclet_seven = "speclet-seven"


@unique
class SlurmPartitions(str, Enum):
    """Partitions of the HPC available through SLURM."""

    priority = "priority"
    interactive = "interactive"
    short = "short"
    medium = "medium"
    long = "long"


#### ---- Models ---- ####


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    model: ModelOption
    fit_method: ModelFitMethod
