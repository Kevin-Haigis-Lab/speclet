#!/usr/bin/env python3

"""Funnctions for handling common modifications and processing of the Achilles data."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.data_processing import common as dphelp

#### ---- Data manipulation ---- ####


def zscale_cna_by_group(
    df: pd.DataFrame,
    gene_cn_col: str = "gene_cn",
    new_col: str = "gene_cn_z",
    groupby_cols: Optional[Union[List[str], Tuple[str, ...]]] = ("hugo_symbol",),
    cn_max: Optional[float] = None,
) -> pd.DataFrame:
    """Z-scale the copy number values.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        gene_cn_col (str, optional): Column with the gene copy number values.
          Defaults to "gene_cn".
        new_col (str, optional): The name of the column to store the calculated values.
          Defaults to "gene_cn_z".
        groupby_cols (Optional[Union[List[str], Tuple[str, ...]]], optional): A list or
          tuple of columns to group the DataFrame by. If None, the rows are not grouped.
          Defaults to ("hugo_symbol").
        cn_max (Optional[float], optional): The maximum copy number to use.
          Defaults to None.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if cn_max is not None and cn_max > 0:
        df[new_col] = df[gene_cn_col].apply(lambda x: np.min((x, cn_max)))
    else:
        df[new_col] = df[gene_cn_col]

    def z_scale(x: pd.Series) -> pd.Series:
        return (x - np.mean(x)) / np.std(x)

    if groupby_cols is not None:
        df[new_col] = df.groupby(list(groupby_cols))[new_col].apply(z_scale)
    else:
        df[new_col] = df[new_col].apply(z_scale)

    return df


#### ---- Indices ---- ####


def make_sgrna_to_gene_mapping_df(
    data: pd.DataFrame, sgrna_col: str = "sgrna", gene_col: str = "hugo_symbol"
) -> pd.DataFrame:
    """Generate a DataFrame mapping sgRNAs to genes.

    Args:
        data (pd.DataFrame): The data set.
        sgrna_col (str, optional): The name of the column with sgRNA data.
          Defaults to "sgrna".
        gene_col (str, optional): The name of the column with gene names.
          Defaults to "hugo_symbol".

    Returns:
        pd.DataFrame: A DataFrame mapping sgRNAs to genes.
    """
    return (
        data[[sgrna_col, gene_col]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(sgrna_col)
        .reset_index(drop=True)
    )


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
    """Object to hold common indicies used for modeling Achilles data."""

    sgrna_idx: np.ndarray
    n_sgrnas: int = 0
    sgrna_to_gene_map: pd.DataFrame
    sgrna_to_gene_idx: np.ndarray
    gene_idx: np.ndarray
    n_genes: int = 0
    cellline_idx: np.ndarray
    n_celllines: int = 0
    kras_mutation_idx: np.ndarray
    n_kras_mutations: int = 0
    batch_idx: np.ndarray
    n_batches: int = 0

    def __init__(self, **data):
        """Object to hold common indicies used for modeling Achilles data."""
        super().__init__(**data)
        self.n_sgrnas = dphelp.nunique(self.sgrna_idx)
        self.n_genes = dphelp.nunique(self.gene_idx)
        self.n_celllines = dphelp.nunique(self.cellline_idx)
        self.n_kras_mutations = dphelp.nunique(self.kras_mutation_idx)
        self.n_batches = dphelp.nunique(self.batch_idx)

    class Config:
        """Configuration for pydantic validation."""

        arbitrary_types_allowed = True


def common_indices(achilles_df: pd.DataFrame, min_kras_muts: int = 0) -> CommonIndices:
    """Generate a collection of indices frequently used when modeling the Achilles data.

    Args:
        achilles_df (pd.DataFrame): The DataFrame with Achilles data.

    Returns:
        Dict[str, Union[np.ndarray, pd.DataFrame]]: A dict with a collection of indices.
    """
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(achilles_df)
    kras_idx = make_kras_mutation_index_with_other(achilles_df, min=min_kras_muts)
    return CommonIndices(
        sgrna_idx=dphelp.get_indices(achilles_df, "sgrna"),
        sgrna_to_gene_map=sgrna_to_gene_map,
        sgrna_to_gene_idx=dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol"),
        gene_idx=dphelp.get_indices(achilles_df, "hugo_symbol"),
        cellline_idx=dphelp.get_indices(achilles_df, "depmap_id"),
        kras_mutation_idx=kras_idx,
        batch_idx=dphelp.get_indices(achilles_df, "pdna_batch"),
    )


#### ---- Data frames ---- ####


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: Union[List[str], Tuple[str, ...]] = (
        "hugo_symbol",
        "depmap_id",
        "sgrna",
        "lineage",
        "chromosome",
        "pdna_batch",
        "kras_mutation",
    ),
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    """Set the appropriate columns of the Achilles data as factors.

    Args:
        data (pd.DataFrame): Achilles DataFrame.
        cols (Union[List[str], Tuple[str, ...]], optional): The names of the columns to
          make categorical. Defaults to [ "hugo_symbol", "depmap_id", "sgrna",
          "lineage", "chromosome", "pdna_batch", ].
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

    data["log2_cn"] = np.log2(data.gene_cn + 1)
    data = zscale_cna_by_group(
        data,
        gene_cn_col="log2_cn",
        new_col="z_log2_cn",
        groupby_cols=["depmap_id"],
        cn_max=np.log2(10),
    )
    data["is_mutated"] = dphelp.nmutations_to_binary_array(data.n_muts)

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
