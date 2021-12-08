"""Paths and data input/output."""

import os
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import Union


@unique
class DataFile(Enum):
    """Data file names."""

    DEPMAP_CRC = "depmap_modeling_dataframe_crc.csv"
    DEPMAP_CRC_BONE = "depmap_modeling_dataframe_crc_bone.csv"
    DEPMAP_CRC_SUBSAMPLE = "depmap_modeling_dataframe_crc-subsample.csv"
    DEPMAP_CRC_BONE_SUBSAMPLE = "depmap_modeling_dataframe_crc_bone-subsample.csv"
    DEPMAP_DATA = "depmap_modeling_dataframe.csv"
    DEPMAP_ESSENTIALS = "known_essentials.csv"
    ACHILLES_GENE_EFFECT = "achilles_gene_effect.csv"
    CCLE_MUTATIONS = "ccle_mutations.csv"
    CCLE_COPYNUMBER = "ccle_gene_cn.csv"
    COPY_NUMBER_SAMPLE = "copy_number_data_samples.npy"
    CGC = "sanger_cancer-gene-census.csv"
    SCREEN_READ_COUNT_TOTALS = "depmap_replicate_total_read_counts.csv"
    PDNA_READ_COUNT_TOTALS = "depmap_pdna_total_read_counts.csv"


def package_root() -> Path:
    """Path to the 'speclet' package root directory.

    Returns:
        Path: Path to the package's root directory.
    """
    return Path(__file__).parent


def project_root() -> Path:
    """Path of the root of the project.

    First looking for the environment variable "PROJECT_ROOT" and falls back to the
    current working directory.

    Returns:
        Path: Path to the root of the project.
    """
    if p := os.getenv("PROJECT_ROOT"):
        return Path(p)
    warnings.warn(
        "No project root dir found ('PROJECT_ROOT') - using current working dir."
    )
    return Path(os.getcwd())


def models_dir() -> Path:
    """Directory for Bayesian model results.

    Returns:
        Path: Path to a directory for storing Bayesian model output.
    """
    if p := os.getenv("MODELS_DIR"):
        return Path(p)
    return project_root() / "models"


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


def modeling_data_dir() -> Path:
    """Path to modeling data directory.

    Returns:
        Path: Path to the modeling data directory.
    """
    return project_root() / "modeling_data"


def stan_models_dir() -> Path:
    """Path to the directory with the Stan model code.

    Returns:
        Path: Directory with the Stan model code.
    """
    return package_root() / "bayesian_models" / "stan_model_code"
