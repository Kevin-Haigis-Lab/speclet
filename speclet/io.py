"""Paths and data input/output."""

import os
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import Final, Union


@unique
class DataFile(Enum):
    """Data file names."""

    DEPMAP_CRC = "DEPMAP_CRC"
    DEPMAP_CRC_BONE = "DEPMAP_CRC_BONE"
    DEPMAP_CRC_SUBSAMPLE = "DEPMAP_CRC_SUBSAMPLE"
    DEPMAP_CRC_BONE_SUBSAMPLE = "DEPMAP_CRC_BONE_SUBSAMPLE"
    DEPMAP_DATA = "DEPMAP_DATA"
    DEPMAP_ESSENTIALS = "DEPMAP_ESSENTIALS"
    DEPMAP_TEST_DATA = "DEPMAP_TEST_DATA"
    ACHILLES_GENE_EFFECT = "ACHILLES_GENE_EFFECT"
    CCLE_MUTATIONS = "CCLE_MUTATIONS"
    CCLE_COPYNUMBER = "CCLE_COPYNUMBER"
    COPY_NUMBER_SAMPLE = "COPY_NUMBER_SAMPLE"
    CGC = "CGC"
    SCREEN_READ_COUNT_TOTALS = "SCREEN_READ_COUNT_TOTALS"
    PDNA_READ_COUNT_TOTALS = "PDNA_READ_COUNT_TOTALS"


_data_file_map: Final[dict[DataFile, str]] = {
    DataFile.DEPMAP_CRC: "depmap_modeling_dataframe_crc.csv",
    DataFile.DEPMAP_CRC_BONE: "depmap_modeling_dataframe_crc_bone.csv",
    DataFile.DEPMAP_CRC_SUBSAMPLE: "depmap_modeling_dataframe_crc-subsample.csv",
    DataFile.DEPMAP_CRC_BONE_SUBSAMPLE: "depmap_modeling_dataframe_crc_bone-subsample.csv",  # noqa: E501
    DataFile.DEPMAP_DATA: "depmap_modeling_dataframe.csv",
    DataFile.DEPMAP_ESSENTIALS: "known_essentials.csv",
    DataFile.DEPMAP_TEST_DATA: "depmap_test_data.csv",
    DataFile.ACHILLES_GENE_EFFECT: "achilles_gene_effect.csv",
    DataFile.CCLE_MUTATIONS: "ccle_mutations.csv",
    DataFile.CCLE_COPYNUMBER: "ccle_gene_cn.csv",
    DataFile.COPY_NUMBER_SAMPLE: "copy_number_data_samples.npy",
    DataFile.CGC: "sanger_cancer-gene-census.csv",
    DataFile.SCREEN_READ_COUNT_TOTALS: "depmap_replicate_total_read_counts.csv",
    DataFile.PDNA_READ_COUNT_TOTALS: "depmap_pdna_total_read_counts.csv",
}


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
        to (DataFile): Either a string that is the name of the desired file or the
        DataFile enum that represents a data file.

    Returns:
        Path: Path to the file.
    """
    if isinstance(to, str):
        return modeling_data_dir() / to

    if to is DataFile.DEPMAP_TEST_DATA:
        _dir = project_root() / "tests"
    else:
        _dir = modeling_data_dir()
    return _dir / _data_file_map[to]


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
