#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import typer

from src.models.crc_ceres_mimic_one import CrcCeresMimicOne
from src.models.crc_model_one import CrcModelOne

app = typer.Typer()


class ModelOption(str, Enum):
    crc_model_one = "crc_model_one"
    crc_ceres_mimic_one = "crc_ceres_mimic_one"


@app.command()
def run_sbc(model_name: ModelOption, cache_dir: Path, perm_number: int) -> None:
    if model_name == ModelOption.crc_model_one:
        model = CrcModelOne(f"sbc{perm_number}", root_cache_dir=cache_dir, debug=True)

    elif model_name == ModelOption.crc_ceres_mimic_one:
        model = CrcCeresMimicOne(
            f"sbc{perm_number}", root_cache_dir=cache_dir, debug=True
        )
    else:
        raise Exception(f"Unknown model '{model_name}'")

    model.run_simulation_based_calibration(
        cache_dir, random_seed=perm_number, size="small"
    )
    return None


if __name__ == "__main__":
    app()
