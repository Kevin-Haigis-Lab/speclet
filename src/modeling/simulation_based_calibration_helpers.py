"""Helpers for organizing simualtaion-based calibrations."""

import math
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arviz as az
import numpy as np
import pandas as pd

from src.data_processing import achilles as achelp
from src.data_processing import vectors as vhelp
from src.io.data_io import DataFile, data_path
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
        _ = self._check_dir_exists()

    def _check_dir_exists(self) -> bool:
        if self.dir.exists():
            return True
        else:
            self.dir.mkdir()
            return False

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


@unique
class SelectionMethod(str, Enum):
    """Methods for selecting `n` elements from a list."""

    random = "random"
    tiled = "tiled"
    repeated = "repeated"
    shuffled = "shuffled"


def select_n_elements_from_l(
    n: int, list: Union[List[Any], np.ndarray], method: Union[SelectionMethod, str]
) -> np.ndarray:
    """Select `n` elements from a collection `l` using a specified method.

    There are three available methods:

    1. `random`: Randomly select `n` values from `l`.
    2. `tiled`: Use `numpy.tile()` (`[1, 2, 3]` → `[1, 2, 3, 1, 2, 3, ...]`).
    3. `repeated`: Use `numpy.repeat()` (`[1, 2, 3]` → `[1, 1, 2, 2, 3, 3, ...]`).
    4. `shuffled`: Shuffles the results of `numpy.tile()` to get even, random coverage.

    Args:
        n (int): Number elements to draw.
        l (Union[List[Any], np.ndarray]): Collection to draw from.
        method (SelectionMethod): Method to use for drawing elements.

    Raises:
        ValueError: Raised if an unknown method is passed.

    Returns:
        np.ndarray: A numpy array of length 'n' with values from 'l'.
    """
    if isinstance(method, str):
        method = SelectionMethod(method)

    size = math.ceil(n / len(list))

    if method == SelectionMethod.random:
        return np.random.choice(list, n)
    elif method == SelectionMethod.tiled:
        return np.tile(list, size)[:n]
    elif method == SelectionMethod.repeated:
        return np.repeat(list, size)[:n]
    elif method == SelectionMethod.shuffled:
        a = np.tile(list, size)[:n]
        np.random.shuffle(a)
        return a
    else:
        raise ValueError(f"Unknown selection method: {method}")


def generate_mock_cell_line_information(
    genes: Union[List[str], np.ndarray],
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool = False,
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
        randomness (bool, optional): Should the lineages, screens, and batches be
          randomly assigned or applied in a pattern? Defaults to False (patterned).

    Returns:
        pd.DataFrame: The mock sample information.
    """
    # Methods for selecting elements from the list to produce pairings.
    _lineage_method = "random" if randomness else "tiled"
    _batch_method = "random" if randomness else "shuffled"
    _screen_method = "random" if randomness else "tiled"

    cell_lines = prefixed_count("cellline", n=n_cell_lines)
    lineages = prefixed_count("lineage", n=n_lineages)
    batches = prefixed_count("batch", n=n_batches)
    batch_map = pd.DataFrame(
        {
            "depmap_id": cell_lines,
            "lineage": select_n_elements_from_l(
                n_cell_lines, lineages, _lineage_method
            ),
            "p_dna_batch": select_n_elements_from_l(
                n_cell_lines, batches, _batch_method
            ),
        }
    )

    screens = prefixed_count("screen", n=n_screens)
    screen_map = pd.DataFrame(
        {
            "p_dna_batch": batches,
            "screen": select_n_elements_from_l(n_batches, screens, _screen_method),
        }
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


def generate_mock_achilles_categorical_groups(
    n_genes: int,
    n_sgrnas_per_gene: int,
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool = False,
) -> pd.DataFrame:
    """Generate mock Achilles categorical column scaffolding.

    This function should be used to generate a scaffolding of the Achilles data. It
    creates columns that mimic the hierarchical natrue of the Achilles categorical
    columns. Each sgRNA maps to a single gene. Each cell lines only received on pDNA
    batch. Each cell line / sgRNA combination occurs exactly once.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNAs per gene.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.
        randomness (bool, optional): Should the lineages, screens, and batches be
          randomly assigned or applied in a pattern? Defaults to False (patterned).

    Returns:
        pd.DataFrame: A pandas data frame the resembles the categorical column
        hierarchical structure of the Achilles data.
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
        randomness=randomness,
    )

    def _make_cat_cols(_df: pd.DataFrame) -> pd.DataFrame:
        return achelp.set_achilles_categorical_columns(_df, cols=_df.columns.tolist())

    return (
        cell_line_info.merge(sgnra_map, on="hugo_symbol")
        .reset_index(drop=True)
        .pipe(_make_cat_cols)
    )


def add_mock_copynumber_data(mock_df: pd.DataFrame) -> pd.DataFrame:
    """Add mock copy number data to mock Achilles data.

    The mock CNA values actually come from real copy number values from CRC cancer cell
    lines. The values are randomly sampled with replacement and some noise is added to
    each value.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.

    Returns:
        pd.DataFrame: Same mock Achilles data frame with a new "copy_number" column.
    """
    real_cna_values = np.load(data_path(DataFile.copy_number_sample))
    mock_cn = np.random.choice(real_cna_values, size=mock_df.shape[0], replace=True)
    mock_cn = mock_cn + np.random.normal(0, 0.1, size=mock_cn.shape)
    mock_cn = vhelp.squish_array(mock_cn, lower=0.0, upper=np.inf)
    mock_df["copy_number"] = mock_cn.flatten()
    return mock_df


def add_mock_rna_expression_data(
    mock_df: pd.DataFrame, groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """Add fake RNA expression data to a mock Achilles data frame.

    The RNA expression values are sampled from a normal distribution with mean and
    standard deviation that are each sampled from different normal distributions. If a
    grouping is supplied, then each value in the group will be sampled from the same
    distribution (i.e. same mean and standard deviation).

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        groups (Optional[List[str]], optional): List of columns to group by. Each group
          will have the same mean and standard deviation for the sampling distribution.
          Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """

    def _rna_normal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        mu = np.abs(np.random.normal(10.0, 3))
        sd = np.abs(np.random.normal(0.0, 3))
        rna_expr = np.random.normal(mu, sd, size=df.shape[0])
        rna_expr = vhelp.squish_array(rna_expr, lower=0.0, upper=np.inf)
        df["rna_expr"] = rna_expr
        return df

    if groups is None:
        mock_df = _rna_normal_distribution(mock_df)
    else:
        mock_df = mock_df.groupby(groups).apply(_rna_normal_distribution)
    return mock_df


def add_mock_is_mutated_data(mock_df: pd.DataFrame, prob: float = 0.01) -> pd.DataFrame:
    """Add a mutation column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        prob (float, optional): The probability of a gene being mutated. All mutations
          are indpendent of each other. Defaults to 0.01.

    Returns:
        pd.DataFrame: The same mock Achilles data frame with an "is_mutated" columns.
    """
    mock_df["is_mutated"] = np.random.uniform(0, 1, size=mock_df.shape[0]) < prob
    return mock_df


def add_mock_zero_effect_lfc_data(
    mock_df: pd.DataFrame, mu: float = 0.0, sigma: float = 0.5
) -> pd.DataFrame:
    """Add fake log-fold change column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        mu (float, optional): Mean of normal distribution for sampling LFC values.
          Defaults to 0.0.
        sigma (float, optional): Standard deviation of normal distribution for sampling
          LFC values. Defaults to 0.5.

    Returns:
        pd.DataFrame: Same mock Achilles data with a new "lfc" column.
    """
    mock_df["lfc"] = np.random.normal(mu, sigma, mock_df.shape[0])
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
    return (
        generate_mock_achilles_categorical_groups(
            n_genes=n_genes,
            n_sgrnas_per_gene=n_sgrnas_per_gene,
            n_cell_lines=n_cell_lines,
            n_lineages=n_lineages,
            n_batches=n_batches,
            n_screens=n_screens,
        )
        .pipe(add_mock_copynumber_data)
        .pipe(add_mock_rna_expression_data, groups=["hugo_symbol", "lineage"])
        .pipe(add_mock_is_mutated_data)
        .pipe(add_mock_zero_effect_lfc_data)
    )
