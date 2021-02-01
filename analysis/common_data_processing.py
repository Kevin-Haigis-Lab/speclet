from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def zscale_cna_by_group(
    df: pd.DataFrame,
    gene_cn_col: str = "gene_cn",
    new_col: str = "gene_cn_z",
    groupby: List[str] = ["hugo_symbol"],
    cn_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Z-scale the copy number values.

    Parameters
    ----------
    df: pandas.DataFrame
        The data
    gene_cn_col: str
        Column name of copy number data
    new_col: str
        Name of new column
    groupby: [str]
        List of columns to group by (using `pandas.groupby(...)`)
    cn_max: num
        Maximum limits for CN values (`None` applies no cap)

    Returns
    -------
    pandas.DataFrame
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

    Parameters
    ----------
    df: pandas.DataFrame
        The data
    col: str
        The column to turn into Categorical
    ordered: bool
        Should the column be ordered? (see `?pandas.Categorical`)
    sort_cats: bool
        Should the list of unique categories be sorted?

    Returns
    -------
    pandas.DataFrame
    """
    categories = df[col].unique().tolist()
    if sort_cats:
        categories.sort()
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Get a list of the indices for a categorical column.

    Parameters
    ----------
    df: pandas.DataFrame
        The data
    col: str
        The column to get indices from

    Returns
    -------
    numpy.ndarray
    """
    return df[col].cat.codes.to_numpy()


def get_indices_and_count(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, int]:
    """
    Get a list of the indices and number of unique values for a categorical column.

    Parameters
    ----------
    df: pandas.DataFrame
        The data
    col: str
        The column to get indices from

    Returns
    -------
    Tuple[numpy.ndarray, int]
    """
    return get_indices(df=df, col=col), df[col].nunique()


def extract_flat_ary(s: pd.Series) -> np.ndarray:
    """
    Turn a column of a DataFrame into a flat array.

    Parameters
    ----------
    df: pandas.Series
        Data to be flattened.

    Returns
    -------
    numpy.ndarray
    """
    return s.to_numpy().flatten()


def nmutations_to_binary_array(m: pd.Series) -> np.ndarray:
    """
    Turn a column of a DataFrame into a binary array of 0's and 1's.

    Parameters
    ----------
    df: pandas.Series
        A column of values to be turned into values of 0 and 1.

    Returns
    -------
    numpy.ndarray
    """
    return extract_flat_ary(m).astype(bool).astype(int)


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: List[str] = ["hugo_symbol", "depmap_id", "sgrna", "lineage", "chromosome"],
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    for col in cols:
        data = make_cat(data, col, ordered=ordered, sort_cats=sort_cats)
    return data


def read_achilles_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)

    data = data.sort_values(
        ["hugo_symbol", "sgrna", "lineage", "depmap_id"]
    ).reset_index(drop=True)

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
