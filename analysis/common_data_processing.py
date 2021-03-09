#!/usr/bin/env python3

import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_errors


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


def nunique(x: Iterable[Any]) -> int:
    """
    Number of unique values in an iterable object.
    """
    if isinstance(x, dict):
        raise TypeError("Cannot count the number of unique values in a dict.")
    return len(np.unique(x))
