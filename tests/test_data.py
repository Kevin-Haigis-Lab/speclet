# Test on the data to ensure it is structured as expected.

import os
from pathlib import Path
from typing import Final

import dask.dataframe as dd
import pandas as pd
import pytest
from dask.distributed import Client

from src.io.data_io import DataFile, data_path

SKIP_DATA_TESTS = os.getenv("DATA_TESTS") is None
FULL_DEPMAP_DATASET_PATH: Final = data_path(DataFile.achilles_data)


@pytest.fixture(scope="module")
def setup_dask() -> Client:
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
    yield client
    client.close()


@pytest.mark.skipif(SKIP_DATA_TESTS, reason="Skip data tests.")
def test_depmap_data_columns_exist(depmap_test_data: Path):
    df = pd.read_csv(depmap_test_data)
    expected_cols = (
        "sgrna",
        "hugo_symbol",
        "depmap_id",
        "replicate_id",
        "lfc",
        "counts_final",
        "p_dna_batch",
        "copy_number",
        "replicate_id",
        "screen",
        "rna_expr",
        "num_mutations",
        "is_mutated",
        "lineage",
        "primary_or_metastasis",
    )
    for col in expected_cols:
        assert col in df.columns


@pytest.mark.skipif(SKIP_DATA_TESTS, reason="Skip data tests.")
def test_depmap_data_no_missing(setup_dask: Client):
    dask_df: dd.DataFrame = dd.read_csv(
        FULL_DEPMAP_DATASET_PATH,
        dtype={
            "age": "float64",
            "p_dna_batch": "object",
            "primary_or_metastasis": "object",
            "counts_final": "float64",
        },
        low_memory=False,
    )
    cols_without_na = [
        "depmap_id",
        "sgrna",
        "hugo_symbol",
        "lfc",
        "screen",
        "num_mutations",
        "is_mutated",
        "lineage",
    ]

    na_checks: pd.Series = dask_df.head().isna()[cols_without_na].any().compute()
    for column, any_missing in na_checks.iteritems():
        assert not any_missing and isinstance(column, str)
