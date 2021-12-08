"""Helpers for biological stuff."""


import janitor  # noqa: F401
import pandas as pd

_chr_cat: list[str] = [str(i) for i in range(1, 22)] + ["X", "Y", "M"]


def make_chromosome_categorical(
    df: pd.DataFrame, chr_col: str = "chr", drop_unused: bool = False
) -> pd.DataFrame:
    """Set a column of chromosome data as categorical.

    Args:
        df (pd.DataFrame): Genomic data.
        chr_col (str, optional): Name of column with chromosome label. Defaults to
          "chr".
        drop_unused (bool, optional): Should chromosomes not present in the data be
          dropped from the categories? Defaults to False.

    Returns:
        pd.DataFrame: The same data frame with the chromosome column of categorical
        data type.
    """
    chr_cats = _chr_cat.copy()
    if drop_unused:
        used_chrs = set(df[chr_col].unique().tolist())
        chr_cats = [c for c in chr_cats if c in used_chrs]

    df[chr_col] = pd.Categorical(
        df[chr_col].values.astype(str), categories=chr_cats, ordered=True
    )
    return df


def extract_chromosome_location_to_df(
    df: pd.DataFrame, col_name: str = "genome_alignment"
) -> pd.DataFrame:
    """Extract information from chromosome location string.

    Break up `chr2_130522105_-` into `chr: "2", pos: 130522105, strand: -1`.

    Args:
        df (pd.DataFrame): Genomic data.
        col_name (str, optional): Column containing the genomic data in as concatenated
          strings. Defaults to "genome_alignment".

    Returns:
        pd.DataFrame: The data frame with three new columns: `"chr"`, `"pos"`, and
        `"strand"`.
    """
    split_info = df[col_name].str.split("_")
    split_info_dict = pd.DataFrame()
    split_info_dict["chr"] = pd.Series(
        [info[0].replace("chr", "") for info in split_info], dtype=str
    )
    split_info_dict["pos"] = pd.Series([info[1] for info in split_info]).astype(int)

    strand_dict: dict[str, int] = {"-": -1, "+": 1}
    split_info_dict["strand"] = pd.Series(
        [strand_dict[info[2]] for info in split_info], dtype=int
    )
    split_info_dict[col_name] = df[col_name]
    split_info_df = pd.DataFrame(split_info_dict)
    return df.merge(split_info_df, on=col_name)
