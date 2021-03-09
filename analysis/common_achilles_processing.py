#!/usr/bin/env python3

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pretty_errors

#### ---- Data manipulation ---- ####


def zscale_cna_by_group(
    df: pd.DataFrame,
    gene_cn_col: str = "gene_cn",
    new_col: str = "gene_cn_z",
    groupby: List[str] = ["hugo_symbol"],
    cn_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Z-scale the copy number values.
    """

    if not cn_max is None and cn_max > 0:
        df[new_col] = df[gene_cn_col].apply(lambda x: np.min((x, cn_max)))
    else:
        df[new_col] = df[gene_cn_col]

    df[new_col] = df.groupby(groupby)[new_col].apply(
        lambda x: (x - np.mean(x)) / np.std(x)
    )

    return df


#### ---- Indices ---- ####


def make_sgrna_to_gene_mapping_df(
    data: pd.DataFrame, sgrna_col: str = "sgrna", gene_col: str = "hugo_symbol"
) -> pd.DataFrame:
    return (
        data[[sgrna_col, gene_col]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(sgrna_col)
        .reset_index(drop=True)
    )


def common_indices(
    achilles_df: pd.DataFrame,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(achilles_df)
    return {
        "sgrna_idx": dphelp.get_indices(achilles_df, "sgrna"),
        "sgrna_to_gene_map": sgrna_to_gene_map,
        "sgrna_to_gene_idx": dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol"),
        "cellline_idx": dphelp.get_indices(achilles_df, "depmap_id"),
        "batch_idx": dphelp.get_indices(achilles_df, "pdna_batch"),
    }


#### ---- Data frames ---- ####


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: List[str] = [
        "hugo_symbol",
        "depmap_id",
        "sgrna",
        "lineage",
        "chromosome",
        "pdna_batch",
    ],
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    """
    Set the appropriate columns of the Achilees data as factors.
    """
    for col in cols:
        data = dphelp.make_cat(data, col, ordered=ordered, sort_cats=sort_cats)
    return data


def read_achilles_data(
    data_path: Path, low_memory: bool = True, set_categorical_cols: bool = True
) -> pd.DataFrame:
    """
    Read in an Achilles data set.
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
        groupby=["depmap_id"],
        cn_max=np.log2(10),
    )
    data["is_mutated"] = dphelp.nmutations_to_binary_array(data.n_muts)

    return data


def subsample_achilles_data(
    df: pd.DataFrame, n_genes: Optional[int] = 100, n_cell_lines: Optional[int] = None
) -> pd.DataFrame:
    """
    Subsample an Achilles data set to a number of genes and/or cell lines.
    """

    if not n_genes is None and n_genes <= 0:
        raise ValueError("Number of genes must be positive.")
    if not n_cell_lines is None and n_cell_lines <= 0:
        raise Exception("Number of genes must be positive.", ValueError)

    genes: List[str] = df.hugo_symbol.values
    cell_lines: List[str] = df.depmap_id.values

    if not n_genes is None:
        genes = np.random.choice(genes, n_genes, replace=False)

    if not n_cell_lines is None:
        cell_lines = np.random.choice(cell_lines, n_cell_lines, replace=False)

    sub_df = df.copy()
    sub_df = sub_df[sub_df.hugo_symbol.isin(genes)]
    sub_df = sub_df[sub_df.depmap_id.isin(cell_lines)]
    return sub_df
