#!/usr/bin/env python3

"""Functions for modifying the everyday pandas DataFrame."""

import warnings
from typing import Any, Collection

import janitor  # noqa: F401
import numpy as np
import pandas as pd

from speclet.loggers import logger


def make_cat(
    df: pd.DataFrame, col: str, ordered: bool = True, sort_cats: bool = False
) -> pd.DataFrame:
    """Make a column of a DataFrame categorical.

    Args:
        df (pd.DataFrame): DataFrame to modify.
        col (str): Column to make categorical.
        ordered (bool, optional): Are the categories ordered? Defaults to True.
        sort_cats (bool, optional): Should the categories be sorted first?
          Defaults to False.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    categories: list[Any] = list(df[col].unique())
    if sort_cats:
        categories.sort()
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    """Get the indices for a categorical column.

    Args:
        df (pd.DataFrame): DataFrame to get an index from.
        col (str): The column containing the index.

    Returns:
        np.ndarray: An array containing the index.
    """
    return df[col].cat.codes.to_numpy()


def get_cats(df: pd.DataFrame, col: str) -> list[str]:
    """Get the categories from a categorical column.

    Args:
        df (pd.DataFrame): Pandas data frame.
        col (str): Column name.

    Returns:
        list[str]: List of categories.
    """
    return df[col].cat.categories.to_list()


def get_indices_and_count(df: pd.DataFrame, col: str) -> tuple[np.ndarray, int]:
    """Get the indices and number of unique values for a categorical column.

    Args:
        df (pd.DataFrame): DataFrame to get an index from.
        col (str): The column containing the index.

    Returns:
        Tuple[np.ndarray, int]: Both the index and number of unique
          values in the column.
    """
    return get_indices(df=df, col=col), df[col].nunique()


def extract_flat_ary(s: pd.Series) -> np.ndarray:
    """Turn a column of a DataFrame into a flat array (deprecated).

    Args:
        s (pd.Series): A series to flatten and convert to a numpy array.

    Returns:
        np.ndarray: The flattened numpy array.
    """
    warnings.warn("Use `df.values` instead of `extract_flat_ary()` 🤦🏻‍♂️", UserWarning)
    return np.asarray(s.values)


def nmutations_to_binary_array(m: pd.Series) -> np.ndarray:
    """Turn a column of a DataFrame into a binary array of 0's and 1's.

    Args:
        m (pd.Series): A pandas Series with mutation information that only
          needs to be binary (is mutated/is not mutated).

    Returns:
        np.ndarray: A logical array.
    """
    if len(m) == 0:
        return np.array([], dtype=int)
    return m.values.astype(bool).astype(int)


def nunique(x: Collection[Any]) -> int:
    """Count the number of unique values in an iterable object.

    Args:
        x (Collection[Any]): The collection of items.

    Returns:
        int: The number of unique items in the input.
    """
    if isinstance(x, dict):
        return len(x)
    return len(set(x))


def center_column_grouped_dataframe(
    df: pd.DataFrame, grp_col: str | list[str], val_col: str, new_col_name: str
) -> pd.DataFrame:
    """Center the values of a column after grouping by another.

    Args:
        df (pd.DataFrame): Pandas DataFrame.
        grp_col (Union[str, List[str]]): The column(s) to group by.
        val_col (str): COlumn with values to center.
        new_col_name (str): New column name to hold the centered values.

    Returns:
        pd.DataFrame: The same input data frame with one new column.
    """
    avg_val_col = f"_avg_{val_col}"
    df_avgs: pd.DataFrame = df.copy()

    if any(df[val_col].isna()):
        logger.warn("There are missing values in data frame; removed from average.")
        df_avgs = df_avgs.loc[~df_avgs[val_col].isna()]

    df_avgs = (
        df_avgs.groupby(grp_col)[val_col]
        .mean()
        .reset_index(drop=False)
        .rename(columns={val_col: avg_val_col})
    )

    return (
        df.merge(df_avgs, how="left", on=grp_col, left_index=False, right_index=False)
        .assign(_new_centered_column=lambda d: d[val_col] - d[avg_val_col])
        .rename(columns={"_new_centered_column": new_col_name})
        .drop(columns=[avg_val_col])
    )


def dataframe_to_matrix(
    df: pd.DataFrame, rows: str, cols: str, values: str, sort_cats: bool = True
) -> np.ndarray:
    """Create a matrix from two nominal columns of a data frame.

    Args:
        df (pd.DataFrame): Input data frame.
        rows (str): Data frame column to place along the rows.
        cols (str): Date fraome column to place along the columns.
        values (str): Values for the matrix.

    Returns:
        np.ndarray: Matrix of shape [`rows`, `cols`].
    """
    if sort_cats:
        sort_cols = [c for c in (rows, cols) if (df[c].dtype.name == "category")]
        if len(sort_cols) > 0:
            df = df.copy().sort_values(sort_cols)

    return (
        df.pivot_wider(index=rows, names_from=cols, values_from=values)
        .drop(columns=[rows])
        .values
    )


def head_tail(
    df: pd.DataFrame,
    n: int = 10,
    n_head: int | None = None,
    n_tail: int | None = None,
) -> pd.DataFrame:
    """Get the 'head' and 'tail' of a data frame.

    Args:
        df (pd.DataFrame): Data frame.
        n (int, optional): Number of rows from both top and bottom. Defaults to 10.
        n_head (Optional[int], optional): Number of rows from the top. Defaults to None.
        n_tail (Optional[int], optional): Number of rows from the bottom. Defaults to
        None.

    Returns:
        pd.DataFrame: 'Head' and 'tail' of the input data frame.
    """
    if n_head is None:
        n_head = n
    if n_tail is None:
        n_tail = n
    return pd.concat([df.head(n_head), df.tail(n_tail)])
