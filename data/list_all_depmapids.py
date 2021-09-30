#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer()


class NoDepMapIDsError(BaseException):
    """No DepMapIDs found."""

    pass


def _get_depmapids(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].dropna().astype(str).unique().tolist()


def get_depmapids_from_achilles(file_path: Path) -> list[str]:
    return _get_depmapids(pd.read_csv(file_path), "DepMap_ID")


def _get_all_replicates_from_read_counts(
    dir: Path, recursive: bool = True
) -> list[str]:
    suffix = ".read_count.tsv.gz"
    read_ct_files: list[str] = []
    for p in dir.iterdir():
        if recursive and p.is_dir():
            read_ct_files += _get_all_replicates_from_read_counts(
                p, recursive=recursive
            )
        elif suffix in p.name:
            read_ct_files.append(p.name)
    return [fname.replace(suffix, "") for fname in read_ct_files]


def get_depmapids_from_score(
    file_path: Path, reads_dir: Optional[Path] = None
) -> list[str]:
    if reads_dir is None:
        return _get_depmapids(pd.read_csv(file_path), "DepMap_ID")

    replicates_with_read_counts = _get_all_replicates_from_read_counts(reads_dir)
    rep_map = (
        pd.read_csv(file_path)
        .drop_duplicates()
        .dropna(subset=["DepMap_ID", "replicate_ID"])
    )
    return _get_depmapids(
        rep_map[rep_map["replicate_ID"].isin(replicates_with_read_counts)],
        "DepMap_ID",
    )


@app.command()
def collect_depmap_ids_from_replicate_maps(
    outfile: Path,
    achilles: Optional[Path] = None,
    score: Optional[Path] = None,
    score_reads_dir: Optional[Path] = None,
    check_any_ids: bool = True,
) -> None:
    """Get DepMap IDs from replicate map files and save as a list.

    Args:
        outfile (Path): Output CSV file with a single `depmap_id` column.
        achilles (Optional[Path], optional): Achilles replicate map. Defaults to None.
        score (Optional[Path], optional): Project Score replicate map. Defaults to
          None.
        score_reads_dir (Optional[Path], optional): Directory containing the separate
          read count files for Project Score data. If provided, this will be used to
          filter the DepMap IDs for Project Score. Defaults to None.
        check_any_ids (bool, optional): Check that each replicate map returned DepMap
          IDs? Defaults to True.

    Raises:
        NoDepMapIDsError: [description]
    """
    depmap_ids: list[str] = []

    if achilles is not None:
        _achilles_ids = get_depmapids_from_achilles(achilles)
        if check_any_ids and len(_achilles_ids) == 0:
            raise NoDepMapIDsError("No IDs found for Achilles.")
        depmap_ids += _achilles_ids

    if score is not None:
        _score_ids = get_depmapids_from_score(score, reads_dir=score_reads_dir)
        if check_any_ids and len(_score_ids) == 0:
            raise NoDepMapIDsError("No IDs found for Project Score.")
        depmap_ids += _score_ids

    depmap_ids = list(set(depmap_ids))
    depmap_ids.sort()
    pd.DataFrame({"depmap_id": depmap_ids}).to_csv(outfile, index=False)


if __name__ == "__main__":
    app()
