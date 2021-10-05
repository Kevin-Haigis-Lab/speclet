#!/usr/bin/env python3

"""Data IO for the speclet project."""

from enum import Enum
from pathlib import Path
from typing import Union

#### ---- Enums ---- ####


class DataFile(str, Enum):
    """Data file names."""

    # crc_data = "depmap_CRC_data.csv"
    DEPMAP_CRC = "depmap_modeling_dataframe_crc.csv"
    DEPMAP_CRC_SUBSAMPLE = "depmap_modeling_dataframe_crc-subsample.csv"
    DEPMAP_DATA = "depmap_modeling_dataframe.csv"
    DEPMAP_ESSENTIALS = "known_essentials.csv"
    ACHILLES_GENE_EFFECT = "achilles_gene_effect.csv"
    CCLE_MUTATIONS = "ccle_mutations.csv"
    CCLE_COPYNUMBER = "ccle_gene_cn.csv"
    COPY_NUMBER_SAMPLE = "copy_number_data_samples.npy"
    CGC = "sanger_cancer-gene-census.csv"
    SCREEN_READ_COUNT_TOTALS = "depmap_replicate_total_read_counts.csv"
    PDNA_READ_COUNT_TOTALS = "depmap_pdna_total_read_counts.csv"


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
