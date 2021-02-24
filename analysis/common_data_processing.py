import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_errors


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


def make_cat(
    df: pd.DataFrame, col: str, ordered: bool = True, sort_cats: bool = False
) -> pd.DataFrame:
    """
    Make a column of a data frame into categorical.
    """
    categories: List[Any] = df[col].unique().tolist()
    if sort_cats:
        categories.sort()
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Get a list of the indices for a categorical column.
    """
    return df[col].cat.codes.to_numpy()


def get_indices_and_count(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, int]:
    """
    Get a list of the indices and number of unique values for a categorical column.
    """
    return get_indices(df=df, col=col), df[col].nunique()


def extract_flat_ary(s: pd.Series) -> np.ndarray:
    """
    Turn a column of a DataFrame into a flat array.
    """
    warnings.warn("Use `df.values` instead of `extract_flat_ary()` 🤦🏻‍♂️", UserWarning)
    return s.values


def nmutations_to_binary_array(m: pd.Series) -> np.ndarray:
    """
    Turn a column of a DataFrame into a binary array of 0's and 1's.
    """
    if len(m) == 0:
        return np.array([], dtype=int)
    return m.values.astype(bool).astype(int)


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: List[str] = ["hugo_symbol", "depmap_id", "sgrna", "lineage", "chromosome"],
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    """
    Set the appropriate columns of the Achilees data as factors.
    """
    for col in cols:
        data = make_cat(data, col, ordered=ordered, sort_cats=sort_cats)
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
    data["is_mutated"] = nmutations_to_binary_array(data.n_muts)

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


def nunique(x: Iterable[Any]) -> int:
    """
    Number of unique values in an iterable object.
    """
    if isinstance(x, dict):
        raise TypeError("Cannot count the number of unique values in a dict.")
    return len(np.unique(x))
