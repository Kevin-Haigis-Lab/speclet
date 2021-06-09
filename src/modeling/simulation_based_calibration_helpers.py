"""Helpers for organizing simualtaion-based calibrations."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arviz as az
import numpy as np
import pandas as pd

from src.data_processing import achilles as achelp
from src.string_functions import prefixed_count


class MockDataSizes(str, Enum):
    """Options for dataset seizes when generating mock data."""

    small = "small"
    medium = "medium"
    large = "large"


class SBCResults:
    """Results from a single round of SBC."""

    priors: Dict[str, Any]
    inference_obj: az.InferenceData
    posterior_summary: pd.DataFrame

    def __init__(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ):
        """Create an instance of SBCResults.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        self.priors = priors
        self.inference_obj = inference_obj
        self.posterior_summary = posterior_summary


class SBCFileManager:
    """Manages the results from a round of simulation-based calibration."""

    dir: Path
    inference_data_path: Path
    priors_path_set: Path
    priors_path_get: Path
    posterior_summary_path: Path

    sbc_results: Optional[SBCResults] = None

    def __init__(self, dir: Path):
        """Create a SBCFileManager.

        Args:
            dir (Path): The directory where the data is stored.
        """
        self.dir = dir
        self.inference_data_path = dir / "inference-data.netcdf"
        self.priors_path_set = dir / "priors"
        self.priors_path_get = dir / "priors.npz"
        self.posterior_summary_path = dir / "posterior-summary.csv"

    def save_sbc_results(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ) -> None:
        """Save the results from a round of SBC.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        inference_obj.to_netcdf(self.inference_data_path.as_posix())
        np.savez(self.priors_path_set.as_posix(), **priors)
        posterior_summary.to_csv(
            self.posterior_summary_path.as_posix(), index_label="parameter"
        )

    def _tidy_numpy_files(self, files: Any) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        for k in files.files:
            d[k] = files[k]
        return d

    def get_sbc_results(self, re_read: bool = False) -> SBCResults:
        """Retrieve results of a round of SBC.

        Args:
            re_read (bool, optional): Should the results be re-read from file?
              Defaults to False.

        Returns:
            SBCResults: The results from the round of SBC.
        """
        if self.sbc_results is not None and not re_read:
            return self.sbc_results

        inference_obj = az.from_netcdf(self.inference_data_path)
        priors_files = np.load(self.priors_path_get.as_posix())
        priors = self._tidy_numpy_files(priors_files)
        posterior_summary = pd.read_csv(
            self.posterior_summary_path, index_col="parameter"
        )

        self.sbc_results = SBCResults(
            priors=priors,
            inference_obj=inference_obj,
            posterior_summary=posterior_summary,
        )
        return self.sbc_results

    def all_data_exists(self) -> bool:
        """Confirm that all data exists.

        Returns:
            bool: True if all of the data exists, else false.
        """
        for p in [
            self.priors_path_get,
            self.posterior_summary_path,
            self.inference_data_path,
        ]:
            if not p.exists():
                return False
        return True


def generate_mock_sgrna_gene_map(n_genes: int, n_sgrnas_per_gene: int) -> pd.DataFrame:
    """Generate a fake sgRNA-gene map.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNA per gene.

    Returns:
        pd.DataFrame: A data frame mapping each sgRNA to a gene. Each sgRNA only matches
          to a single gene and each gene will have `n_sgrnas_per_gene` sgRNAs mapped
          to it.
    """
    genes = prefixed_count("gene", n=n_genes)
    sgrnas = [prefixed_count(gene + "_sgrna", n=n_sgrnas_per_gene) for gene in genes]
    return pd.DataFrame(
        {
            "hugo_symbol": np.repeat(genes, n_sgrnas_per_gene),
            "sgrna": np.array(sgrnas).flatten(),
        }
    )


def generate_mock_cell_line_information(
    genes: Union[List[str], np.ndarray],
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
) -> pd.DataFrame:
    """Generate mock "sample information" for fake cell lines.

    Args:
        genes (List[str]): List of genes tested in the cell lines.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.

    Returns:
        pd.DataFrame: The mock sample information.
    """
    cell_lines = prefixed_count("cellline", n=n_cell_lines)
    lineages = prefixed_count("lineage", n=n_lineages)
    batches = prefixed_count("batch", n=n_batches)
    batch_map = pd.DataFrame(
        {
            "depmap_id": cell_lines,
            "lineage": np.random.choice(lineages, n_cell_lines),
            "p_dna_batch": np.random.choice(batches, n_cell_lines),
        }
    )

    screens = prefixed_count("screen", n=n_screens)
    screen_map = pd.DataFrame(
        {"p_dna_batch": batches, "screen": np.random.choice(screens, n_batches)}
    )
    return (
        pd.DataFrame(
            {
                "depmap_id": np.repeat(cell_lines, len(np.unique(genes))),
                "hugo_symbol": np.tile(genes, n_cell_lines),
            }
        )
        .merge(batch_map, on="depmap_id")
        .merge(screen_map, on="p_dna_batch")
    )


def add_mock_copynumber_data(mock_df: pd.DataFrame) -> pd.DataFrame:
    """Add mock copy number data to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.

    Returns:
        pd.DataFrame: Same mock Achilles data frame with a new "copy_number" column.
    """
    mock_df["copy_number"] = 2 ** np.random.normal(1, 0.5, mock_df.shape[0])
    return mock_df


def add_mock_zero_effect_lfc_data(mock_df: pd.DataFrame) -> pd.DataFrame:
    """Add fake log-fold change column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.

    Returns:
        pd.DataFrame: Same mock Achilles data with a new "lfc" column.
    """
    mock_df["lfc"] = np.random.normal(0, 0.5, mock_df.shape[0])
    return mock_df


def generate_mock_achilles_data(
    n_genes: int,
    n_sgrnas_per_gene: int,
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
) -> pd.DataFrame:
    """Generate mock Achilles data.

    Each sgRNA maps to a single gene. Each cell lines only received on pDNA batch.
    Each cell line / sgRNA combination occurs exactly once.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNAs per gene.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.

    Returns:
        pd.DataFrame: A pandas data frame the resembles the Achilles data.
    """
    sgnra_map = generate_mock_sgrna_gene_map(
        n_genes=n_genes, n_sgrnas_per_gene=n_sgrnas_per_gene
    )
    cell_line_info = generate_mock_cell_line_information(
        genes=sgnra_map.hugo_symbol.unique(),
        n_cell_lines=n_cell_lines,
        n_lineages=n_lineages,
        n_batches=n_batches,
        n_screens=n_screens,
    )

    def _make_cat_cols(_df: pd.DataFrame) -> pd.DataFrame:
        return achelp.set_achilles_categorical_columns(_df, cols=_df.columns.tolist())

    df = (
        cell_line_info.merge(sgnra_map, on="hugo_symbol")
        .reset_index(drop=True)
        .pipe(_make_cat_cols)
        .pipe(add_mock_copynumber_data)
        .pipe(add_mock_zero_effect_lfc_data)
    )

    return df
