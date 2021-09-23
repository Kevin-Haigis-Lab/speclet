#!/usr/bin/env python3

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import typer

app = typer.Typer()


class NoDepMapIDsError(BaseException):
    """No DepMapIDs found."""

    pass


def _get_depmapids(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].astype(str).unique().tolist()


def get_depmapids_from_achilles(file_path: Path) -> list[str]:
    return _get_depmapids(pd.read_csv(file_path), "DepMap_ID")


def get_depmapids_from_score(file_path: Path) -> list[str]:
    return _get_depmapids(pd.read_csv(file_path), "DepMap_ID")


@app.command()
def collect_depmap_ids_from_replicate_maps(
    outfile: Path,
    achilles: Optional[Path] = None,
    score: Optional[Path] = None,
    check_any_ids: bool = True,
) -> None:
    """Get DepMap IDs from replicate map files and save as a list.

    Args:
        outfile (Path): Output CSV file with a single `depmap_id` column.
        achilles (Optional[Path], optional): Achilles replicate map. Defaults to None.
        score (Optional[Path], optional): Project Score replicate map. Defaults to
          None.
        check_any_ids (bool, optional): Check that each replicate map returned DepMap
          IDs? Defaults to True.

    Raises:
        NoDepMapIDsError: [description]
    """
    depmap_ids: list[str] = []

    file_to_function: dict[Optional[Path], Callable[[Path], list[str]]] = {
        achilles: get_depmapids_from_achilles,
        score: get_depmapids_from_score,
    }

    for fp, fxn in file_to_function.items():
        if fp is None:
            continue

        ids = fxn(fp)
        if check_any_ids and len(ids) == 0:
            raise NoDepMapIDsError(f"No IDs found for '{fp}'.")

        depmap_ids += ids

    depmap_ids = list(set(depmap_ids))
    depmap_ids.sort()
    pd.DataFrame({"depmap_id": depmap_ids}).to_csv(outfile, index=False)


if __name__ == "__main__":
    app()
