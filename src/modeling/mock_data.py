"""For the noble and fraught persuit of the faithful generation of mock data."""

import math
from enum import Enum, unique
from random import choices
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.data_processing import achilles as achelp
from src.data_processing import vectors as vhelp
from src.io.data_io import DataFile, data_path
from src.string_functions import prefixed_count


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
    sgrna_target_chr = choices(["Chr1", "Chr2", "Chr3"], k=n_genes)
    sgrnas = [prefixed_count(gene + "_sgrna", n=n_sgrnas_per_gene) for gene in genes]
    return pd.DataFrame(
        {
            "hugo_symbol": np.repeat(genes, n_sgrnas_per_gene),
            "sgrna_target_chr": np.repeat(sgrna_target_chr, n_sgrnas_per_gene),
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
    n: int, list: Union[list[Any], np.ndarray], method: Union[SelectionMethod, str]
) -> np.ndarray:
    """Select `n` elements from a collection `l` using a specified method.

    There are three available methods:

    1. `random`: Randomly select `n` values from `l`.
    2. `tiled`: Use `numpy.tile()` (`[1, 2, 3]` → `[1, 2, 3, 1, 2, 3, ...]`).
    3. `repeated`: Use `numpy.repeat()` (`[1, 2, 3]` → `[1, 1, 2, 2, 3, 3, ...]`).
    4. `shuffled`: Shuffles the results of `numpy.tile()` to get even, random coverage.

    Args:
        n (int): Number elements to draw.
        l (Union[list[Any], np.ndarray]): Collection to draw from.
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
    genes: Union[list[str], np.ndarray],
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool = False,
) -> pd.DataFrame:
    """Generate mock "sample information" for fake cell lines.

    Args:
        genes (list[str]): List of genes tested in the cell lines.
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

    screens = ["broad"]
    if n_screens == 2:
        screens += ["sanger"]
    if n_screens > 2:
        screens += prefixed_count("screen", n=n_screens - 2)

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


def _make_mock_grouped_copy(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]]
) -> pd.DataFrame:
    df_copy = mock_df.copy()
    if grouping_cols is not None:
        df_copy = df_copy[grouping_cols].drop_duplicates()
    return df_copy


def _merge_mock_and_grouped_copy(
    mock_df: pd.DataFrame, df_copy: pd.DataFrame, grouping_cols: Optional[list[str]]
) -> pd.DataFrame:
    if grouping_cols is not None:
        return mock_df.merge(df_copy, left_index=False, right_index=False)
    return df_copy


def add_mock_copynumber_data(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Add mock copy number data to mock Achilles data.

    The mock CNA values actually come from real copy number values from CRC cancer cell
    lines. The values are randomly sampled with replacement and some noise is added to
    each value.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same CN value. Defaults to
          None for no group effect.

    Returns:
        pd.DataFrame: Same mock Achilles data frame with a new "copy_number" column.
    """
    real_cna_values = np.load(data_path(DataFile.copy_number_sample))

    df_copy = _make_mock_grouped_copy(mock_df, grouping_cols)
    mock_cn = np.random.choice(real_cna_values, size=df_copy.shape[0], replace=True)
    mock_cn = mock_cn + np.random.normal(0, 0.1, size=mock_cn.shape)
    mock_cn = vhelp.squish_array(mock_cn, lower=0.0, upper=20.0)
    df_copy["copy_number"] = mock_cn.flatten()
    return _merge_mock_and_grouped_copy(mock_df, df_copy, grouping_cols)


def add_mock_rna_expression_data(
    mock_df: pd.DataFrame,
    grouping_cols: Optional[list[str]] = None,
    subgroups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add fake RNA expression data to a mock Achilles data frame.

    The RNA expression values are sampled from a normal distribution with mean and
    standard deviation that are each sampled from different normal distributions. If a
    grouping is supplied, then each value in the group will be sampled from the same
    distribution (i.e. same mean and standard deviation).

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same RNA value. Defaults to
          None for no group effect.
        subgroups (Optional[list[str]], optional): List of columns to group by. Each
          group will have the same mean and standard deviation for the sampling
          distribution. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """

    def _rna_normal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = _make_mock_grouped_copy(df, grouping_cols)
        mu = np.abs(np.random.normal(10.0, 3))
        sd = np.abs(np.random.normal(0.0, 3))
        rna_expr = np.random.normal(mu, sd, size=df_copy.shape[0])
        rna_expr = vhelp.squish_array(rna_expr, lower=0.0, upper=np.inf)
        df_copy["rna_expr"] = rna_expr
        df_copy = _merge_mock_and_grouped_copy(df, df_copy, grouping_cols)
        return df_copy

    if subgroups is None:
        mock_df = _rna_normal_distribution(mock_df)
    else:
        mock_df = (
            mock_df.groupby(subgroups)
            .apply(_rna_normal_distribution)
            .reset_index(drop=True)
        )
    return mock_df


def add_mock_is_mutated_data(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]] = None, prob: float = 0.01
) -> pd.DataFrame:
    """Add a mutation column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same mutation value. Defaults
          to None for no group effect.
        prob (float, optional): The probability of a gene being mutated. All mutations
          are indpendent of each other. Defaults to 0.01.

    Returns:
        pd.DataFrame: The same mock Achilles data frame with an "is_mutated" columns.
    """
    df_copy = _make_mock_grouped_copy(mock_df, grouping_cols)
    df_copy["is_mutated"] = np.random.uniform(0, 1, size=df_copy.shape[0]) < prob
    return _merge_mock_and_grouped_copy(mock_df, df_copy, grouping_cols)


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
        .pipe(add_mock_copynumber_data, grouping_cols=["hugo_symbol", "depmap_id"])
        .pipe(
            add_mock_rna_expression_data,
            grouping_cols=["hugo_symbol", "depmap_id"],
            subgroups=["hugo_symbol", "lineage"],
        )
        .pipe(add_mock_is_mutated_data, grouping_cols=["hugo_symbol", "depmap_id"])
        .pipe(add_mock_zero_effect_lfc_data)
    )
