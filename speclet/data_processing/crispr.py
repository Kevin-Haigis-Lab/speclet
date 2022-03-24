"""Functions for handling common modifications and processing of the Achilles data."""

from pathlib import Path
from typing import Final, Iterable, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from speclet.data_processing import common as dphelp
from speclet.data_processing.vectors import careful_zscore, squish_array
from speclet.io import DataFile, data_path
from speclet.loggers import logger

#### ---- Data manipulation ---- ####


def zscale_cna_by_group(
    df: pd.DataFrame,
    cn_col: str = "copy_number",
    new_col: str = "copy_number_z",
    groupby_cols: Optional[Union[list[str], tuple[str, ...]]] = ("hugo_symbol",),
    cn_max: Optional[float] = None,
    center: Optional[float] = None,
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
        center (Optional[float], optional): The value to use for the center. If `None`
        (default), the average is used.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if cn_max is not None and cn_max > 0:
        df[new_col] = squish_array(df[cn_col].values, lower=0, upper=cn_max)
    else:
        df[new_col] = df[cn_col]

    def zscore_cna_col(d: pd.DataFrame) -> pd.DataFrame:
        _avg = d[new_col].mean()
        d[new_col] = careful_zscore(d[new_col].values)
        if center is not None:
            d[new_col] = d[new_col] + _avg - center
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

    rna_z = careful_zscore(rna, atol=0.01)

    if lower_bound is not None and upper_bound is not None:
        rna_z = squish_array(rna_z, lower=lower_bound, upper=upper_bound)

    df[new_col] = rna_z
    return df


ArgToZscaleByExpression = Union[str, Optional[str], Optional[float]]


def zscale_rna_expression_by_gene_lineage(
    df: pd.DataFrame,
    rna_col: str = "rna_expr",
    *args: ArgToZscaleByExpression,
    **kwargs: ArgToZscaleByExpression,
) -> pd.DataFrame:
    """Z-scale RNA expression data grouping by lineage and gene.

    All positional and keyword arguments are passed to `zscale_rna_expression()`.

    Args:
        df (pd.DataFrame): The Achilles data frame.

    Returns:
        pd.DataFrame: The original data frame with a new column with the z-scaled RNA
          expression values.
    """
    rna_expr_df = (
        df.copy()[["hugo_symbol", "lineage", "depmap_id", rna_col]]
        .drop_duplicates()
        .groupby(["hugo_symbol", "lineage"])
        .apply(zscale_rna_expression, rna_col=rna_col, *args, **kwargs)
        .drop(columns=[rna_col])
    )
    return df.merge(rna_expr_df, how="left", on=["hugo_symbol", "lineage", "depmap_id"])


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
        .pipe(dphelp.make_cat, col=kg, sort_cats=True)
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

    def __init__(self, **data: Union[int, np.ndarray, pd.DataFrame]):
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

    def __init__(self, **data: Union[int, np.ndarray, pd.DataFrame]):
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


#### ---- Data frames ---- ####

_default_achilles_categorical_cols: Final[tuple[str, ...]] = (
    "hugo_symbol",
    "depmap_id",
    "sgrna",
    "lineage",
    "sgrna_target_chr",
    "p_dna_batch",
    "screen",
)


def set_achilles_categorical_columns(
    data: pd.DataFrame,
    cols: Iterable[str] = _default_achilles_categorical_cols,
    ordered: bool = True,
    sort_cats: bool = False,
) -> pd.DataFrame:
    """Set the appropriate columns of the Achilles data as factors.

    If the column is not actually in the data frame, it is just skipped.

    Args:
        data (pd.DataFrame): Achilles DataFrame.
        cols (Union[List[str], Tuple[str, ...]], optional): The names of the columns to
          make categorical. Defaults to ("hugo_symbol", "depmap_id", "sgrna",
          "lineage", "sgrna_target_chr", "p_dna_batch", "screen").
        ordered (bool, optional): Should the categorical columns be ordered?
          Defaults to True.
        sort_cats (bool, optional): Should the categorical columns be sorted?
          Defaults to False.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    for col in cols:
        if col in data.columns:
            data = dphelp.make_cat(data, col, ordered=ordered, sort_cats=sort_cats)
    return data


def sort_achilles_data(data: pd.DataFrame) -> pd.DataFrame:
    """Sort a CRISPR screen data frame.

    Args:
        data (pd.DataFrame): CRISPR screen data frame.

    Returns:
        pd.DataFrame: Sorted data frame.
    """
    return data.sort_values(
        ["hugo_symbol", "sgrna", "lineage", "depmap_id"]
    ).reset_index(drop=True)


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
    data = pd.read_csv(data_path, low_memory=low_memory).pipe(sort_achilles_data)

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

    genes: np.ndarray = df.hugo_symbol.unique()
    cell_lines: np.ndarray = df.depmap_id.unique()

    if n_genes is not None:
        genes = np.random.choice(genes, n_genes, replace=False)

    if n_cell_lines is not None:
        cell_lines = np.random.choice(cell_lines, n_cell_lines, replace=False)

    sub_df: pd.DataFrame = df.copy()
    sub_df = sub_df[sub_df.hugo_symbol.isin(genes)]
    sub_df = sub_df[sub_df.depmap_id.isin(cell_lines)]
    return sub_df


def append_total_read_counts(
    achilles_df: pd.DataFrame,
    final_reads_total: Optional[Path] = None,
    p_dna_reads_total: Optional[Path] = None,
    final_reads_total_colname: str = "counts_final_total",
    initial_reads_total_colname: str = "counts_initial_total",
) -> pd.DataFrame:
    """Append columns with total read count data.

    Args:
        achilles_df (pd.DataFrame): Achilles data frame.
        final_reads_total (Optional[Path], optional): Path to the final read totals
          table (as a CSV). Defaults to None.
        p_dna_reads_total (Optional[Path], optional): Path to the initial read totals
          table (as a CSV). Defaults to None.
        final_reads_total_colname (str, optional): Name for the column with the total
          number of final read counts. Defaults to "counts_final_total".
        initial_reads_total_colname (str, optional): Name for the column with the total
          number of initial read counts. Defaults to "counts_initial_total".

    Returns:
        pd.DataFrame: The initial data frame with two new columns.
    """
    if final_reads_total is None:
        final_reads_total = data_path(DataFile.SCREEN_READ_COUNT_TOTALS)
    if p_dna_reads_total is None:
        p_dna_reads_total = data_path(DataFile.PDNA_READ_COUNT_TOTALS)

    final_reads_total_df = pd.read_csv(final_reads_total)
    p_dna_reads_total_df = pd.read_csv(p_dna_reads_total)

    return (
        achilles_df.merge(final_reads_total_df, on="replicate_id")
        .rename(columns={"total_reads": final_reads_total_colname})
        .merge(p_dna_reads_total_df, on="p_dna_batch")
        .rename(columns={"total_reads": initial_reads_total_colname})
    )


def add_useful_read_count_columns(
    crispr_df: pd.DataFrame,
    counts_final: str = "counts_final",
    counts_final_total: str = "counts_final_total",
    counts_initial: str = "counts_initial",
    counts_initial_total: str = "counts_initial_total",
    counts_final_rpm: str = "counts_final_rpm",
    counts_initial_adj: str = "counts_initial_adj",
    copy: bool = False,
) -> pd.DataFrame:
    """Add some useful columns for modeling read count data.

    - final counts RPM =
      \\(1^6 \\times (c_\\text{final} / \\Sigma c_\\text{final}) + 1\\)
    - adjusted initial counts = \\((c_\\text{initial} / \\Sigma c_\\text{initial})
      \\times \\Sigma c_\\text{final}\\)

    Args:
        crispr_df (pd.DataFrame): Achilles data frame
        counts_final (str, optional): Column of final read counts. Defaults to
          "counts_final".
        counts_final_total (str, optional): Column of total final read counts. Defaults
          to "counts_final_total".
        counts_initial (str, optional): Column of initial read counts. Defaults to
          "counts_initial".
        counts_initial_total (str, optional): Column of total initial read counts.
          Defaults to "counts_initial_total".
        counts_final_rpm (str, optional): Column name for the new column with the final
          read counts in "reads per million" (RPM). Defaults to "counts_final_rpm".
        counts_initial_adj (str, optional): Column name for the new column with the
          adjusted initial read counts. Defaults to "counts_initial_adj".
        copy (bool, optional): First copy the data frame? Defaults to False.

    Returns:
        pd.DataFrame: The original data frame with the new columns append.
    """
    if copy:
        crispr_df = crispr_df.copy()

    # 1e6 * (counts_f / Σ counts_f) + 1
    crispr_df[counts_final_rpm] = (
        1e6 * (crispr_df[counts_final] / crispr_df[counts_final_total]) + 1
    )
    # (counts_i / Σ counts_i) * Σ counts_f
    crispr_df[counts_initial_adj] = (
        crispr_df[counts_initial] / crispr_df[counts_initial_total]
    ) * crispr_df[counts_final_total]

    return crispr_df


def add_one_to_counts(
    crispr_df: pd.DataFrame, cols: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Add 1 to counts columns.

    Args:
        crispr_df (pd.DataFrame): CRISPR screen data frame.
        cols (Optional[Iterable[str]], optional): Columns to add 1 to. Defaults to
        `counts_final` and `counts_initial_adj`.

    Returns:
        pd.DataFrame: Modified data frame.
    """
    if cols is None:
        cols = ["counts_final", "counts_initial_adj"]
    for col in cols:
        crispr_df[col] = crispr_df[col] + 1
    return crispr_df


def filter_for_broad_source_only(
    df: pd.DataFrame, screen_col: str = "screen"
) -> pd.DataFrame:
    """Filter for only data from the Broad.

    Args:
        df (pd.DataFrame): CRISPR screen data.
        screen_col (str, optional): Name of the column indicating the origin of the
          data. Defaults to "screen".

    Returns:
        pd.DataFrame: The filtered data frame.
    """
    return df[df[screen_col] == "broad"].reset_index(drop=True)


def _get_sgrnas_that_map_to_multiple_genes(
    df: pd.DataFrame, sgrna_col: str
) -> np.ndarray:
    return (
        make_sgrna_to_gene_mapping_df(df)
        .groupby([sgrna_col])["hugo_symbol"]
        .count()
        .reset_index()
        .query("hugo_symbol > 1")[sgrna_col]
        .unique()
    )


def drop_sgrnas_that_map_to_multiple_genes(
    df: pd.DataFrame, sgrna_col: str = "sgrna", gene_col: str = "hugo_symbol"
) -> pd.DataFrame:
    """Drop sgRNAs that map to multiple genes.

    Because of how the multi-hitting sgRNAs are identified, this function in not
    "Dask friendly."

    Args:
        df (pd.DataFrame): CRISPR screen data frame.
        sgrna_col (str, optional): sgRNA column name. Defaults to "sgrna".
        gene_col (str, optional): Gene column name. Defaults to "hugo_symbol".

    Returns:
        pd.DataFrame: The filtered data frame.
    """
    sgrnas_to_remove = _get_sgrnas_that_map_to_multiple_genes(df, sgrna_col)
    logger.warning(
        f"Dropping {len(sgrnas_to_remove)} sgRNA that map to multiple genes."
    )
    df_new = df.copy()[~df[sgrna_col].isin(sgrnas_to_remove)]
    return df_new


def drop_missing_copynumber(
    df: pd.DataFrame, cn_col: str = "copy_number"
) -> pd.DataFrame:
    """From data points with missing copy number data.

    Args:
        df (pd.DataFrame): CRISPR screen data.
        cn_col (str, optional): Name of the column with copy number data. Defaults to
          "copy_number".

    Returns:
        pd.DataFrame: Filtered data frame.
    """
    df_new = df.copy()[~df[cn_col].isna()]
    size_diff = df.shape[0] - df_new.shape[0]
    logger.warning(f"Dropping {size_diff} data points with missing copy number.")
    return df_new
