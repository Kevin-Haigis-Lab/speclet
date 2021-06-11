"""Create auxillary data files for different parts of the project."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import validate_arguments

#### ---- Sample copy number data for use in generating mock data ---- ####


@validate_arguments
def sample_copy_number_data(source_file: Path, out_file: Path) -> None:
    df = pd.read_csv(source_file, low_memory=False)
    cna = np.random.choice(df["copy_number"].values, size=10_000, replace=False)
    np.save(file=out_file, arr=cna, allow_pickel=True, fix_imports=False)


snakemake: Any  # for mypy

sample_copy_number_data(
    source_file=snakemake.input["crc_subset"],  # noqa: F821
    out_file=snakemake.output["cna_sample"],  # noqa: F821
)
