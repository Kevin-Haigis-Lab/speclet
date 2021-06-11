"""Create auxillary data files for different parts of the project."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import validate_arguments

#### ---- Sample copy number data for use in generating mock data ---- ####

np.random.seed(826)


@validate_arguments
def sample_copy_number_data(source_file: Path, out_file: Path) -> None:
    df = pd.read_csv(source_file, low_memory=False)
    cna = df["copy_number"].values
    cna = cna[np.logical_not(np.isnan(cna))]
    cna = np.random.choice(cna, size=50_000, replace=False)
    np.save(file=out_file, arr=cna, allow_pickle=True, fix_imports=False)


snakemake: Any  # for mypy

sample_copy_number_data(
    source_file=snakemake.input["crc_subset"],  # noqa: F821
    out_file=snakemake.output["cna_sample"],  # noqa: F821
)
