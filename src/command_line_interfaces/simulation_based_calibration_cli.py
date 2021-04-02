#!/usr/bin/env python3

"""A helpful copmmand line interface for simulation-based calibration."""

from enum import Enum
from pathlib import Path
from typing import Dict, Type

import typer

from src.models.crc_ceres_mimic_one import CrcCeresMimicOne
from src.models.crc_model_one import CrcModelOne
from src.models.speclet_model import SpecletModel

app = typer.Typer()


class ModelOption(str, Enum):
    """Model options."""

    crc_model_one = "crc_model_one"
    crc_ceres_mimic_one = "crc_ceres_mimic_one"


def get_model_class(model_opt: ModelOption) -> Type[SpecletModel]:
    """Get the model class from its string identifier.

    Args:
        model_opt (ModelOption): The string identifier for the model.

    Returns:
        Type[SpecletModel]: The corresponding model class.
    """
    model_option_map: Dict[ModelOption, Type[SpecletModel]] = {
        ModelOption.crc_model_one: CrcModelOne,
        ModelOption.crc_ceres_mimic_one: CrcCeresMimicOne,
    }
    return model_option_map[model_opt]


@app.command()
def run_sbc(
    model_name: ModelOption, cache_dir: Path, sim_number: int, data_size: str
) -> None:
    """CLI for running a round of simulation-based calibration for a model.

    Args:
        model_name (ModelOption): Name of the model to use.
        cache_dir (Path): Where to store the results.
        sim_number (int): Simulation number.
        data_size (str): Which data size to use. See the actual methods for details and options.

    Returns:
        [type]: [description]
    """
    ModelClass = get_model_class(model_opt=model_name)
    model = ModelClass(f"sbc{sim_number}", root_cache_dir=cache_dir, debug=True)
    model.run_simulation_based_calibration(
        cache_dir, random_seed=sim_number, size=data_size
    )
    return None


if __name__ == "__main__":
    app()
