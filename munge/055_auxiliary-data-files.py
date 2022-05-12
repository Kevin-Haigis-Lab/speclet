#!/usr/bin/env python3

"""Create auxillary data files for different parts of the project."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer()


# --- Sample copy number data for use in generating mock data ---

np.random.seed(826)


@app.command()
def sample_copy_number_data(source_file: Path, out_file: Path) -> None:
    df = pd.read_csv(source_file, low_memory=False)
    cna = df["copy_number"].values
    cna = cna[np.logical_not(np.isnan(cna))]
    cna = np.random.choice(cna, size=50_000, replace=False)
    np.save(file=out_file, arr=cna, allow_pickle=True, fix_imports=False)


if __name__ == "__main__":
    app()
