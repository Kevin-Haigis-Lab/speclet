"""Feature engineering for models."""

import pandas as pd

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.loggers import logger


class ColumnAlreadyExistsError(BaseException):
    """Column already exists error."""

    pass


#### ---- Copy number ---- ####


def centered_copynumber_by_cellline(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column of centered copy number values by cell line.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"copy_number_cellline"`.
    """
    _new_col = "copy_number_cellline"
    logger.info(f"Adding '{_new_col}' column.")
    if _new_col in df.columns:
        raise ColumnAlreadyExistsError(_new_col)
    return dphelp.center_column_grouped_dataframe(
        df,
        grp_col="depmap_id",
        val_col="copy_number",
        new_col_name=_new_col,
    )


def centered_copynumber_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column of centered copy number values by gene.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"copy_number_gene"`.
    """
    _new_col = "copy_number_gene"
    logger.info(f"Adding '{_new_col}' column.")
    if _new_col in df.columns:
        raise ColumnAlreadyExistsError(_new_col)
    return dphelp.center_column_grouped_dataframe(
        df,
        grp_col="hugo_symbol",
        val_col="copy_number",
        new_col_name=_new_col,
    )


#### ---- RNA expression ---- ####


def zscale_rna_expression_by_gene_and_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scale the RNA expression per gene in each lineage.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"rna_expr_gene_lineage"`.
    """
    logger.info("Adding 'rna_expr_gene_lineage' column.")
    return achelp.zscale_rna_expression_by_gene_lineage(
        df,
        rna_col="rna_expr",
        new_col="rna_expr_gene_lineage",
        lower_bound=-5.0,
        upper_bound=5.0,
    )


#### ---- Mutations ---- ####


def convert_is_mutated_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the boolean column for gene mutation status to type integer.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with the column `"is_mutated"` as type integer.
    """
    logger.info("Converting 'is_mutated' column to 'int'.")
    df["is_mutated"] = df["is_mutated"].astype(int)
    return df
