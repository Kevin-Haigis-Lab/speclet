#!/usr/bin/env python3

"""Data IO for the speclet project."""

from enum import Enum
from pathlib import Path
from typing import Union

#### ---- Enums ---- ####


class DataFile(str, Enum):
    """Data file names."""

    # crc_data = "depmap_CRC_data.csv"
    crc_data = "depmap_modeling_dataframe_crc.csv"
    crc_subsample = "depmap_modeling_dataframe_crc-subsample.csv"
    achilles_data = "depmap_modeling_dataframe.csv"
    achilles_subsample = "depmap_modeling_dataframe_crc-subsample.csv"
    achilles_essentials = "known_essentials.csv"
    achilles_gene_effect = "achilles_gene_effect.csv"
    ccle_mutations = "ccle_mutations.csv"
    ccle_copynumber = "ccle_gene_cn.csv"


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


def data_path(to: Union[str, DataFile]) -> Path:
    """Path a to a data file.

    Args:
        to (DataFile): The desired data.

    Returns:
        Path: Path to the file.
    """
    if isinstance(to, str):
        to = DataFile(to)
    return modeling_data_dir() / to.value
