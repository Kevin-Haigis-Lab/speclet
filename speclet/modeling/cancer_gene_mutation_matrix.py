"""Cancer gene mutation matrix preparation."""

from itertools import product
from typing import Callable, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

from speclet.data_processing.common import get_cats, get_indices
from speclet.loggers import logger


def _cell_line_by_cancer_gene_mutation_matrix(
    data: pd.DataFrame,
    cancer_genes: list[str],
    gene_col: str = "hugo_symbol",
    cell_line_col: str = "depmap_id",
    mut_col: str = "is_mutated",
) -> xr.DataArray:
    """Create a binary matrix of [cancer gene x cell line].

    I did this verbosely with a numpy matrix and iteration to make sure I didn't drop
    any cell lines or cancer genes never mutated and to ensure the order of each group.
    """
    if data[cell_line_col].dtype == "category":
        cells = get_cats(data, cell_line_col)
    else:
        cells = data[cell_line_col].unique().tolist()

    mat = np.zeros((len(cells), len(cancer_genes)))
    mutations = (
        data.copy()[[cell_line_col, gene_col, mut_col]]
        .drop_duplicates()
        .filter_column_isin(gene_col, cancer_genes)
        .reset_index(drop=True)
        .astype({mut_col: np.int32})
    )
    for (i, cell), (j, gene) in product(enumerate(cells), enumerate(cancer_genes)):
        _query = f"`{cell_line_col}` == '{cell}' and `{gene_col}` == '{gene}'"
        is_mut = mutations.query(_query)[mut_col]
        assert len(is_mut) == 1
        mat[i, j] = is_mut.values[0]
    mat = mat.astype(np.int32)
    return xr.DataArray(
        mat,
        dims=("cell_line", "cancer_gene"),
        coords={"cell_line": cells, "cancer_gene": cancer_genes},
        name=mut_col,
    )


def _trim_cancer_genes(
    cg_mut_matrix: xr.DataArray,
    min_n: int = 1,
    min_freq: float = 0.0,
) -> xr.DataArray:
    """Trim cancer genes and mutation matrix to avoid colinearities.

    Corrects for:
        1. remove cancer genes never mutated (or above a threshold)
        2. remove cancer genes always mutated
    """
    # Identifying cancer genes to remove.
    all_mut = np.all(cg_mut_matrix, axis=0)
    logger.debug("all_mut: {}")
    low_n_mut = np.sum(cg_mut_matrix, axis=0) < min_n
    low_freq_mut = np.mean(cg_mut_matrix, axis=0) < min_freq
    drop_idx = all_mut + low_n_mut + low_freq_mut

    # Logging.
    _dropped_cancer_genes = list(
        np.asarray(cg_mut_matrix.coords["cancer_gene"])[drop_idx]
    )
    logger.info(f"Dropping {len(_dropped_cancer_genes)} cancer genes.")
    if len(_dropped_cancer_genes) > 0:
        logger.debug(f"Dropped cancer genes: {_dropped_cancer_genes}")

    # Execute changes.
    cg_mut_matrix = cg_mut_matrix[:, ~drop_idx]
    return cg_mut_matrix


def _get_colinear_columns(mat: xr.DataArray, col: int) -> list[int]:
    col_vals = mat[:, col]
    to_merge: list[int] = [col]
    for i in range(mat.shape[1]):
        if i == col:
            continue
        if np.all(col_vals == mat[:, i]):
            to_merge.append(i)
    return to_merge


def _merge_cancer_genes(
    genes: list[str], colinear_cols: list[int], keep_pos: int
) -> list[str]:
    merge_genes = [genes[i] for i in colinear_cols]
    new_gene = "|".join(merge_genes)
    genes[keep_pos] = new_gene
    return [g for g in genes if g not in merge_genes]


def _merge_colinear_columns(
    mat: xr.DataArray, colinear_cols: list[int], keep_pos: int
) -> xr.DataArray:
    cgs: list[str] = mat.coords["cancer_gene"].values.tolist()
    merged_cgs = _merge_cancer_genes(
        cgs, colinear_cols=colinear_cols, keep_pos=keep_pos
    )
    drop_cols = [c for c in colinear_cols if c != keep_pos]
    keep_idx = np.array([i not in drop_cols for i in range(mat.shape[1])])
    mat = mat[:, keep_idx]
    mat.coords["cancer_gene"] = merged_cgs
    return mat


def _n_cancer_genes(m: xr.DataArray) -> int:
    return m.shape[1]


def _merge_colinear_cancer_genes(cg_mut_mat: xr.DataArray) -> xr.DataArray:
    """Merge perfectly comutated cancer genes into a single column."""
    n_cg = _n_cancer_genes(cg_mut_mat) + 1
    while n_cg != _n_cancer_genes(cg_mut_mat) and _n_cancer_genes(cg_mut_mat) > 0:
        n_cg = _n_cancer_genes(cg_mut_mat)
        for i in range(n_cg):
            colinear_cols = _get_colinear_columns(cg_mut_mat, i)
            if len(colinear_cols) > 1:
                cg_mut_mat = _merge_colinear_columns(
                    cg_mut_mat, colinear_cols, keep_pos=i
                )
                break
    return cg_mut_mat


def _top_n_cancer_genes(mut_mat: xr.DataArray, top_n: int) -> xr.DataArray:
    """Select only the top-n most frequently mutated cancer genes."""
    n_muts = mut_mat.sum(axis=0)
    mut_order = rankdata(-1 * n_muts, method="min") - 1
    n_cgs = _n_cancer_genes(mut_mat)
    assert len(n_muts) == n_cgs
    assert len(mut_order) == n_cgs
    keep_idx = mut_order < top_n
    dropped_cgs = mut_mat.coords["cancer_gene"][~keep_idx]
    if len(dropped_cgs) > 0:
        logger.debug(f"Drop {len(dropped_cgs)} cancer genes to get top-{top_n}.")
        logger.debug(f"Dropped: {dropped_cgs}")
        logger.debug(f"`mut_order`: {mut_order}")
        logger.debug(f" `keep_idx`: {keep_idx}")
    return mut_mat[:, keep_idx]


def _transform_to_data_pt_by_cancer_gene(
    cell_x_cg_mat: xr.DataArray, data: pd.DataFrame, cell_line_col: str
) -> xr.DataArray:
    """Convert a [cell x cancer gene] array into [data pt. x cancer gene]."""
    if data[cell_line_col].dtype == "cetegory":
        cell_line_idx = get_indices(data, cell_line_col)
    else:
        cell_line_idx = get_indices(
            data.copy().astype({cell_line_col: "category"}), cell_line_col
        )
    cg_mut_mat = cell_x_cg_mat[cell_line_idx, :].rename({"cell_line": "data_pt"})
    cg_mut_mat.coords["data_pt"] = list(range(len(data)))
    return cg_mut_mat


def _get_cell_lines_with_gene_mutated(
    df: pd.DataFrame, gene: str, gene_col: str, cell_line_col: str, mut_col: str
) -> set[str]:
    """Get the cell lines with a mutation in a specific gene."""
    return set(
        df.query(f"`{gene_col}` == '{gene}' and `{mut_col}`")[cell_line_col].unique()
    )


def _get_cell_lines_with_gene_mutated_memo(
    gene_col: str, cell_line_col: str, mut_col: str
) -> Callable[[pd.DataFrame, str], set[str]]:
    """Provides a memoized wrapper around `_get_cell_lines_with_gene_mutated()`."""
    memory: dict[str, set[str]] = {}

    def _memo_fxn(_df: pd.DataFrame, _gene: str) -> set[str]:
        if _gene in memory:
            return memory[_gene]
        muts = _get_cell_lines_with_gene_mutated(
            df=_df,
            gene=_gene,
            gene_col=gene_col,
            cell_line_col=cell_line_col,
            mut_col=mut_col,
        )
        memory[_gene] = muts
        return muts

    return _memo_fxn


def _remove_comutation_collinearity(
    data: pd.DataFrame,
    cancer_gene_mut_mat: xr.DataArray,
    gene_col: str,
    cell_line_col: str,
    mut_col: str,
) -> xr.DataArray:
    """Remove cases where the target gene is always mutated with a cancer gene.

    Prevents collinearity between the cancer gene comutation variable and target gene
    mutation variable. This collinearity is always present for the cancer genes,
    themselves, and this is automatically handled here, too.
    """
    _cell_lines_with_gene_mut = _get_cell_lines_with_gene_mutated_memo(
        gene_col=gene_col, cell_line_col=cell_line_col, mut_col=mut_col
    )

    for i, cg in enumerate(cancer_gene_mut_mat.coords["cancer_gene"].values):
        cg_muts = _cell_lines_with_gene_mut(data, cg)
        for gene in data[gene_col].unique():
            gene_muts = _cell_lines_with_gene_mut(data, gene)
            if cg_muts == gene_muts:
                gene_idx = data[gene_col] == gene
                cancer_gene_mut_mat[gene_idx, i] = 0
    return cancer_gene_mut_mat


def _drop_cancer_genes_with_no_comutation(cg_mut_mat: xr.DataArray) -> xr.DataArray:
    """Drop any columns and cancer genes with no mutations."""
    any_muts = cg_mut_mat.sum(axis=0) > 0
    if np.all(any_muts):
        return cg_mut_mat
    return cg_mut_mat[:, any_muts]


def make_cancer_gene_mutation_matrix(
    data: pd.DataFrame,
    cancer_genes: list[str],
    gene_col: str = "hugo_symbol",
    cell_line_col: str = "depmap_id",
    mut_col: str = "is_mutated",
    min_n: int = 1,
    min_freq: float = 0.0,
    top_n_cg: int | None = None,
) -> xr.DataArray | None:
    """Make a [data pt. x cancer gene] mutation matrix.

    Args:
        data (pd.DataFrame): Data
        cancer_genes (list[str]): Cancer genes.
        gene_col (str, optional): Name of the column in the data frame with the gene
        names. Defaults to "hugo_symbol".
        cell_line_col (str, optional): Name of the column in the data frame with the
        cell line names. Defaults to "depmap_id".
        mut_col (str, optional): Name of the column in the data frame with a boolean for
        if the target gene is mutated. Defaults to "is_mutated".
        min_n (int, optional): Minimum number of cells lines with a mutation for each
        cancer gene. Defaults to 1.
        min_freq (float, optional): Minimum mutational frequency for each cancer gene.
        Defaults to 0.0.
        top_n_cg (int | None, optional): Select only the top-n most frequently mutated
        cancer genes. Defaults to None.

    Returns:
        xr.DataArray | None: Data array with the shape [data pt. x cancer gene] with a 0
        or 1 for whether data pt. i has a mutation in cancer gene j. If one of the steps
        removes all cancer genes, then `None` is returned.
    """
    if len(cancer_genes) == 0:
        logger.debug("No cancer genes provided.")
        return None

    cell_x_cg_mat = _cell_line_by_cancer_gene_mutation_matrix(
        data=data,
        cancer_genes=cancer_genes,
        gene_col=gene_col,
        cell_line_col=cell_line_col,
        mut_col=mut_col,
    )
    cell_x_cg_mat = _trim_cancer_genes(cell_x_cg_mat, min_n=min_n, min_freq=min_freq)
    if _n_cancer_genes(cell_x_cg_mat) == 0:
        logger.debug("Removed all cancer genes during trimming step.")
        return None

    cell_x_cg_mat = _merge_colinear_cancer_genes(cell_x_cg_mat)
    if top_n_cg is not None:
        cell_x_cg_mat = _top_n_cancer_genes(cell_x_cg_mat, top_n=top_n_cg)

    cg_mut_mat = _transform_to_data_pt_by_cancer_gene(
        cell_x_cg_mat, data, cell_line_col=cell_line_col
    )
    cg_mut_mat = _remove_comutation_collinearity(
        data,
        cg_mut_mat,
        gene_col=gene_col,
        cell_line_col=cell_line_col,
        mut_col=mut_col,
    )
    cg_mut_mat = _drop_cancer_genes_with_no_comutation(cg_mut_mat)
    if _n_cancer_genes(cg_mut_mat) == 0:
        logger.debug("Removed all cancer genes due to correlation with other genes.")
        return None

    return cg_mut_mat


def extract_mutation_matrix_and_cancer_genes(
    cg_mut_mat: xr.DataArray,
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Convert the cancer gene mutation xarray DataArray to a numpy array and list.

    Args:
        cg_mut_mat (xr.DataArray): Cancer gene mutation matrix.

    Returns:
        tuple[npt.NDArray[np.int32], list[str]]: Standard Numpy matrix and the
        corresponding list of cancer genes.
    """
    return cg_mut_mat.values, cg_mut_mat.coords["cancer_gene"].values.tolist()


@overload
def convert_to_dataframe(cg_mut_mat: None) -> None:
    ...


@overload
def convert_to_dataframe(cg_mut_mat: xr.DataArray) -> pd.DataFrame:
    ...


def convert_to_dataframe(cg_mut_mat: xr.DataArray | None) -> pd.DataFrame | None:
    """Convert the cancer gene mutation matrix to a wide data frame.

    Args:
        cg_mut_mat (xr.DataArray | None): Cancer gene mutation matrix.

    Returns:
        pd.DataFrame: Wide data frame of dimensions [data pt. x cancer gene].
    """
    if cg_mut_mat is None:
        return None

    return (
        cg_mut_mat.to_dataframe()
        .reset_index()
        .pivot_wider("data_pt", names_from="cancer_gene", values_from="is_mutated")
        .drop(columns=["data_pt"])
    )
