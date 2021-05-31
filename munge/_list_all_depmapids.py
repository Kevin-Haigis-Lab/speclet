#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


def main(in_path: Path, out_path: Path) -> None:
    _ = (
        pd.read_csv(in_path)
        .rename(columns={"DepMap_ID": "depmap_id"})
        .drop_duplicates()
        .reset_index()
        .to_csv(out_path, index=False)
    )


if __name__ == "__main__":
    main(
        in_path=Path("data", "depmap_20q3", "Achilles_replicate_map.csv"),
        out_path=Path("modeling_data", "all_achilles_depmapids.csv"),
    )
