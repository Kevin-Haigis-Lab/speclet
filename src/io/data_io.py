#!/usr/bin/env python3

"""Data IO for the speclet project."""

from enum import Enum
from pathlib import Path

#### ---- Enums ---- ####


class DataFile(str, Enum):
    """Data file names."""

    crc_data = "depmap_CRC_data.csv"
    crc_subsample = "depmap_CRC_data_subsample.csv"
    achilles_data = "depmap_modeling_dataframe.csv"
    achilles_subsample = "depmap_modeling_dataframe_subsample.csv"


#### ---- Basics ---- ####


def project_root_dir() -> Path:
    """Speclet root directory.

    Returns:
        Path: Path to root directory.
    """
    return Path(__file__).parent.parent.parent


def modeling_data_dir() -> Path:
    """Path to modeling data directory.

    Returns:
        Path: Path to the modeling data directory.
    """
    return project_root_dir() / "modeling_data"


#### ---- Getters ---- ####


def data_path(to: DataFile) -> Path:
    """Path a to a data file.

    Args:
        to (DataFile): The desired data.

    Returns:
        Path: Path to the file.
    """
    return modeling_data_dir() / to.value
