#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

df = Path("data", "depmap_20q3", "Achilles_replicate_map.csv")
data = (
    pd.read_csv(df)
    .rename(columns={"DepMap_ID": "depmap_id"})
    .drop_duplicates()
    .reset_index()
    .to_csv(index=False)
)
