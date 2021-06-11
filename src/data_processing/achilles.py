#!/usr/bin/env python3

"""Functions for handling common modifications and processing of the Achilles data."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.data_processing import common as dphelp
from src.data_processing.vectors import careful_zscore, squish_array

#### ---- Data manipulation ---- ####


def zscale_cna_by_group(
    df: pd.DataFrame,
    cn_col: str = "copy_number",
    new_col: str = "copy_number_z",
    groupby_cols: Optional[Union[List[str], Tuple[str, ...]]] = ("hugo_symbol",),
    cn_max: Optional[float] = None,
) -> pd.DataFrame:
    """Z-scale the copy number values.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        cn_col (str, optional): Column with the gene copy number values.
          Defaults to "copy_number".
        new_col (str, optional): The name of the column to store the calculated values.
          Defaults to "copy_number_z".
        groupby_cols (Optional[Union[List[str], Tuple[str, ...]]], optional): A list or
          tuple of columns to group the DataFrame by. If None, the rows are not grouped.
          Defaults to ("hugo_symbol").
        cn_max (Optional[float], optional): The maximum copy number to use.
          Defaults to None.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if cn_max is not None and cn_max > 0:
        df[new_col] = squish_array(df[cn_col].values, lower=0, upper=cn_max)
    else:
        df[new_col] = df[cn_col]

    def zscore_cna_col(d: pd.DataFrame):
        d[new_col] = careful_zscore(d[new_col].values)
        return d

    if groupby_cols is None:
        df = zscore_cna_col(df)
    else:
        df = df.groupby(list(groupby_cols)).apply(zscore_cna_col)

    return df


def zscale_rna_expression(
    df: pd.DataFrame,
    rna_col: str = "rna_expr",
    new_col: Optional[str] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.DataFrame:
    """Z-scale RNA expression data.

    If there is not enough variation in the values, dividing by the standard deviation
    becomes very unstable. Thus, in this function, if the values are all too similar,
    all are set to 0 instead of either `NaN` or very extreme values.

    Args:
        df (pd.DataFrame): Data frame.
        rna_col (str, optional): Column with RNA expr data. Defaults to "rna_expr".
        new_col (Optional[str], optional): Name of the new column to be generated.
          Defaults to `f"{rna_col}_z"` if None.
        lower_bound (Optional[float], optional): Hard lower bound on the scaled values.
          Defaults to None.
        upper_bound (Optional[float], optional): Hard upper bound on the scaled values.
          Defaults to None.

    Returns:
        pd.DataFrame: The original data frame with a new column with the z-scaled RNA
          expression values.
    """
    if new_col is None:
        new_col = rna_col + "_z"

    rna = df[rna_col].values
    rna_z = careful_zscore(rna, atol=0.01, transform=lambda x: np.log10(x + 1))

    if lower_bound is not None and upper_bound is not None:
        rna_z = squish_array(rna_z, lower=lower_bound, upper=upper_bound)

    df[new_col] = rna_z
    return df


def zscale_rna_expression_by_gene_lineage(
    df: pd.DataFrame, *args, **kwargs
) -> pd.DataFrame:
    """Z-scale RNA expression data grouping by lineage and gene.

    All positional and keyword arguments are passed to `zscale_rna_expression()`.

    Args:
        df (pd.DataFrame): The Achilles data frame.

    Returns:
        pd.DataFrame: The original data frame with a new column with the z-scaled RNA
          expression values.
    """
    return df.groupby(["lineage", "hugo_symbol"]).apply(
        zscale_rna_expression, *args, **kwargs
    )


#### ---- Indices ---- ####


def make_mapping_df(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Generate a DataFrame mapping two columns.

    Args:
        data (pd.DataFrame): The data set.
        col1 (str): The name of the column with the lower level group (group that will
          have all values appear exactly once).
        col2 (str): The name of the column with the higher level group (multiple values
          in group 1 will map to a single value in group 2).

    Returns:
        pd.DataFrame: A DataFrame mapping the values in col1 and col2.
    """
    return (
        data[[col1, col2]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(col1)
        .reset_index(drop=True)
    )


def make_sgrna_to_gene_mapping_df(
    data: pd.DataFrame, sgrna_col: str = "sgrna", gene_col: str = "hugo_symbol"
) -> pd.DataFrame:
    """Generate a DataFrame mapping sgRNAs to genes.

    Args:
        data (pd.DataFrame): The data set.
        sgrna_col (str, optional): The name of the column with sgRNA data. Defaults to
         "sgrna".
        gene_col (str, optional): The name of the column with gene names. Defaults to
          "hugo_symbol".

    Returns:
        pd.DataFrame: A DataFrame mapping sgRNAs to genes.
    """
    return make_mapping_df(data, sgrna_col, gene_col)


def make_cell_line_to_lineage_mapping_df(
    data: pd.DataFrame, cell_line_col: str = "depmap_id", lineage_col: str = "lineage"
) -> pd.DataFrame:
    """Generate a DataFrame mapping cell lines to lineages.

    Args:
        data (pd.DataFrame): The data set.
        cell_line_col (str, optional): The name of the column with cell line names.
          Defaults to "depmap_id".
        lineage_col (str, optional): The name of the column with lineages. Defaults to
          "lineage".

    Returns:
        pd.DataFrame: A DataFrame mapping cell lines to lineages.
    """
    return make_mapping_df(data, cell_line_col, lineage_col)


def make_kras_mutation_index_with_other(
    df: pd.DataFrame,
    min: int = 0,
    kras_col: str = "kras_mutation",
    cl_col: str = "depmap_id",
) -> np.ndarray:
    """KRAS indexing with other for rare mutations.

    Args:
        df (pd.DataFrame): Data frame to make index for.
        min (int, optional): Minimim number of cell lines with the mutation to keep it
          as a separate group. Defaults to 0.
        kras_col (str, optional): Column name with KRAS mutations. Defaults to
          "kras_mutation".
        cl_col (str, optional): Column name with cell line identifiers. Defaults to
          "depmap_id".

    Raises:
        ValueError: Raised if the indicated columns do not exist.

    Returns:
        np.ndarray: Index for KRAS alleles.
    """
    for col in (kras_col, cl_col):
        if col not in df.columns:
            raise ValueError(f"Could not find column '{col}' in data frame.")
    kg = "__kras_group"
    mut_freq = (
        df[[kras_col, cl_col]]
        .drop_duplicates()
        .groupby(kras_col)[[cl_col]]
        .count()
        .reset_index(drop=False)
    )
    mut_freq[kg] = [
        k if n >= min else "__other__"
        for k, n in zip(mut_freq[kras_col], mut_freq[cl_col])
    ]
    mut_freq = mut_freq[[kras_col, kg]]
    return (
        pd.merge(df.copy(), mut_freq, how="left", on=kras_col)
        .pipe(dphelp.make_cat, col=kg)
        .pipe(dphelp.get_indices, col=kg)
    )


class CommonIndices(BaseModel):
    """Object to hold common indices used for modeling Achilles data."""

    sgrna_idx: np.ndarray
    n_sgrnas: int = 0
    sgrna_to_gene_map: pd.DataFrame
    sgrna_to_gene_idx: np.ndarray
    gene_idx: np.ndarray
    n_genes: int = 0
    cellline_idx: np.ndarray
    n_celllines: int = 0
    lineage_idx: np.ndarray
    n_lineages: int = 0
    cellline_to_lineage_map: pd.DataFrame
    cellline_to_lineage_idx: np.ndarray

    def __init__(self, **data):
        """Object to hold common indices used for modeling Achilles data."""
        super().__init__(**data)
        self.n_sgrnas = dphelp.nunique(self.sgrna_idx)
        self.n_genes = dphelp.nunique(self.gene_idx)
        self.n_celllines = dphelp.nunique(self.cellline_idx)
        self.n_lineages = dphelp.nunique(self.lineage_idx)

    class Config:
        """Configuration for pydantic validation."""

        arbitrary_types_allowed = True


def common_indices(achilles_df: pd.DataFrame) -> CommonIndices:
    """Generate a collection of indices frequently used when modeling the Achilles data.

    Args:
        achilles_df (pd.DataFrame): The DataFrame with Achilles data.

    Returns:
        CommonIndices: A data model with a collection of indices.
    """
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(achilles_df)
    cellline_to_lineage_map = make_cell_line_to_lineage_mapping_df(achilles_df)
    return CommonIndices(
        sgrna_idx=dphelp.get_indices(achilles_df, "sgrna"),
        sgrna_to_gene_map=sgrna_to_gene_map,
        sgrna_to_gene_idx=dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol"),
        gene_idx=dphelp.get_indices(achilles_df, "hugo_symbol"),
        cellline_idx=dphelp.get_indices(achilles_df, "depmap_id"),
        lineage_idx=dphelp.get_indices(achilles_df, "lineage"),
        cellline_to_lineage_map=cellline_to_lineage_map,
        cellline_to_lineage_idx=dphelp.get_indices(cellline_to_lineage_map, "lineage"),
        batch_idx=dphelp.get_indices(achilles_df, "p_dna_batch"),
    )


class DataBatchIndices(BaseModel):
    """Object to hold indices relating to data screens and batches."""

    batch_idx: np.ndarray
    n_batches: int = 0
    screen_idx: np.ndarray
    n_screens: int = 0
    batch_to_screen_map: pd.DataFrame
    batch_to_screen_idx: np.ndarray

    def __init__(self, **data):
        """Object to hold indices relating to data screens and batches."""
        super().__init__(**data)
        self.n_batches = dphelp.nunique(self.batch_idx)
        self.n_screens = dphelp.nunique(self.screen_idx)

    class Config:
        """Configuration for pydantic validation."""

        arbitrary_types_allowed = True


def data_batch_indices(achilles_df: pd.DataFrame) -> DataBatchIndices:
    """Generate a collection of indices relating to data screens and batches.

    Args:
        achilles_df (pd.DataFrame): The DataFrame with Achilles data.

    Returns:
        DataBatchIndices: A data model with a collection of indices.
    """
    batch_to_screen_map = make_mapping_df(
        achilles_df, col1="p_dna_batch", col2="screen"
    )
    return DataBatchIndices(
        batch_idx=dphelp.get_indices(achilles_df, "p_dna_batch"),
        screen_idx=dphelp.get_indices(achilles_df, "screen"),
        batch_to_screen_map=batch_to_screen_map,
        batch_to_screen_idx=dphelp.get_indices(batch_to_screen_map, "screen"),
    )


# class UncommonIndices(BaseModel):
#     """Object to hold uncommon indices used for modeling Achilles data."""

#     cellline_to_kras_mutation_idx: np.ndarray
#     n_kras_mutations: int = 0

#     def __init__(self, **data):
#         """Object to hold common indices used for modeling Achilles data."""
#         super().__init__(**data)
#         self.n_kras_mutations = dphelp.nunique(self.cellline_to_kras_mutation_idx)

#     class Config:
#         """Configuration for pydantic validation."""

#         arbitrary_types_allowed = True


# def uncommon_indices(
#     achilles_df: pd.DataFrame, min_kras_muts: int = 0
# ) -> UncommonIndices:
#     """Generate a collection of indices frequently used for modeling Achilles data.

#     Args:
#         achilles_df (pd.DataFrame): The DataFrame with Achilles data.

#     Returns:
#         UncommonIndices: A data model with a collection of indices.
#     """
#     mod_df = achilles_df.copy()[["depmap_id", "kras_mutation"]]
#     mod_df["kras_idx"] = make_kras_mutation_index_with_other(
#         achilles_df, min=min_kras_muts
#     )
#     mod_df = mod_df.drop_duplicates().sort_values("depmap_id").reset_index(drop=True)
#     cl_to_kras_idx = mod_df["kras_idx"].values
#     return UncommonIndices(cellline_to_kras_mutation_idx=cl_to_kras_idx)


#### ---- Data frames ---- ####


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: Union[List[str], Tuple[str, ...]] = (
        "hugo_symbol",
        "depmap_id",
        "sgrna",
        "lineage",
        "sgrna_target_chr",
        "p_dna_batch",
        "screen",
    ),
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    """Set the appropriate columns of the Achilles data as factors.

    Args:
        data (pd.DataFrame): Achilles DataFrame.
        cols (Union[List[str], Tuple[str, ...]], optional): The names of the columns to
          make categorical. Defaults to ("hugo_symbol", "depmap_id", "sgrna",
          "lineage", "sgrna_target_chr", "p_dna_batch", "sreen").
        ordered (bool, optional): Should the categorical columns be ordered?
          Defaults to True.
        sort_cats (bool, optional): Should the categorical columns be sorted?
          Defaults to False.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    for col in cols:
        data = dphelp.make_cat(data, col, ordered=ordered, sort_cats=sort_cats)
    return data


def read_achilles_data(
    data_path: Path, low_memory: bool = True, set_categorical_cols: bool = True
) -> pd.DataFrame:
    """Read in an Achilles data set.

    Args:
        data_path (Path): The path to the data set.
        low_memory (bool, optional): Should pandas be informed of memory constraints?
          Defaults to True.
        set_categorical_cols (bool, optional): Should the default categorical columns
          be set? Defaults to True.

    Returns:
        pd.DataFrame: The Achilles data set.
    """
    data = pd.read_csv(data_path, low_memory=low_memory)

    data = data.sort_values(
        ["hugo_symbol", "sgrna", "lineage", "depmap_id"]
    ).reset_index(drop=True)

    if set_categorical_cols:
        data = set_achilles_categorical_columns(data)

    return data


def subsample_achilles_data(
    df: pd.DataFrame, n_genes: Optional[int] = 100, n_cell_lines: Optional[int] = None
) -> pd.DataFrame:
    """Subsample an Achilles data set to a number of genes and/or cell lines.

    Args:
        df (pd.DataFrame): Achilles data.
        n_genes (Optional[int], optional): Number of genes to subsample.
          Defaults to 100.
        n_cell_lines (Optional[int], optional): Number of cell lines to subsample.
          Defaults to None.

    Raises:
        ValueError: If the number of genes or cell lines is not positive.

    Returns:
        pd.DataFrame: The Achilles data set.
    """
    if n_genes is not None and n_genes <= 0:
        raise ValueError("Number of genes must be positive.")
    if n_cell_lines is not None and n_cell_lines <= 0:
        raise ValueError("Number of cell lines must be positive.")

    genes: List[str] = df.hugo_symbol.unique()
    cell_lines: List[str] = df.depmap_id.unique()

    if n_genes is not None:
        genes = np.random.choice(genes, n_genes, replace=False)

    if n_cell_lines is not None:
        cell_lines = np.random.choice(cell_lines, n_cell_lines, replace=False)

    sub_df: pd.DataFrame = df.copy()
    sub_df = sub_df[sub_df.hugo_symbol.isin(genes)]
    sub_df = sub_df[sub_df.depmap_id.isin(cell_lines)]
    return sub_df
