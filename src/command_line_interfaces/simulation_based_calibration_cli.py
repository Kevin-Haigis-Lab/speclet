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
    """Get the model class from its string identifier."""
    model_option_map: Dict[ModelOption, Type[SpecletModel]] = {
        ModelOption.crc_model_one: CrcModelOne,
        ModelOption.crc_ceres_mimic_one: CrcCeresMimicOne,
    }
    return model_option_map[model_opt]


@app.command()
def run_sbc(
    model_name: ModelOption, cache_dir: Path, perm_number: int, data_size: str
) -> None:
    """CLI for running a round of simulation-based calibration for a model."""
    ModelClass = get_model_class(model_opt=model_name)
    model = ModelClass(f"sbc{perm_number}", root_cache_dir=cache_dir, debug=True)
    model.run_simulation_based_calibration(
        cache_dir, random_seed=perm_number, size=data_size
    )
    return None


if __name__ == "__main__":
    app()
