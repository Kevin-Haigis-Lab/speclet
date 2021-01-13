import numpy as np
import pandas as pd


def zscale_cna_by_group(
    df, gene_cn_col="gene_cn", new_col="gene_cn_z", groupby=["hugo_symbol"], cn_max=None
):
    """
    Z-scale the copy number values.

    Parameters
    ----------
    df: pandas DataFrame
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
    pandas DataFrame
    """

    if not cn_max is None and cn_max > 0:
        df[new_col] = df[gene_cn_col].apply(lambda x: np.min((x, cn_max)))
    else:
        df[new_col] = df[gene_cn_col]

    df[new_col] = df.groupby(groupby)[new_col].apply(
        lambda x: (x - np.mean(x)) / np.std(x)
    )

    return df


def make_cat(df, col, ordered=True, sort_cats=False):
    """
    Make a column of a data frame into categorical.

    Parameters
    ----------
    df: pandas DataFrame
        The data
    col: str
        The column to turn into Categorical
    ordered: bool
        Should the column be ordered? (see `?pandas.Categorical`)
    sort_cats: bool
        Should the list of unique categories be sorted?

    Returns
    -------
    pandas DataFrame
    """
    categories = df[col].unique().tolist()
    if sort_cats:
        categories.sort()
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_indices(df, col):
    """
    Get a list of the indices for a categorical column.

    Parameters
    ----------
    df: pandas DataFrame
        The data
    col: str
        The column to get indices from

    Returns
    -------
    pandas DataFrame
    """
    return df[col].cat.codes.to_numpy()
