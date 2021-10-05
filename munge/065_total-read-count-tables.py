#!/usr/bin/env python3

"""Tables with total read counts for each replicate and pDNA batch."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import typer
from dask.distributed import Client

app = typer.Typer()


@contextmanager
def dask_client(
    n_workers: int = 4, threads_per_worker: int = 4, memory_limit: str = "16GB"
) -> Generator[None, None, None]:
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    try:
        yield client
    finally:
        client.close()


def _build_depmap_modeling_df_totals_table(
    depmap_modeling_df_path: Path,
) -> pd.DataFrame:
    with dask_client() as _:
        screen_data = dd.read_csv(depmap_modeling_df_path, low_memory=False)
        replicate_read_count_totals = (
            screen_data[["replicate_id", "counts_final"]]
            .dropna()
            .astype({"counts_final": "int"})
            .groupby("replicate_id")["counts_final"]
            .sum()
            .compute()
            .reset_index(drop=False)
            .rename(columns={"counts_final": "total_reads"})
        )
    return replicate_read_count_totals


def _build_achilles_pdna_totals_table(achilles_pdna_df_path: Path) -> pd.DataFrame:
    with dask_client() as _:
        achilles_pdna = dd.read_csv(achilles_pdna_df_path, low_memory=False)
        achilles_pdna_count_totals = (
            achilles_pdna[["p_dna_batch", "median_rpm"]]
            .assign(median_rpm_reverted=lambda d: d.median_rpm - 1)
            .groupby("p_dna_batch")["median_rpm_reverted"]
            .sum()
            .compute()
            .reset_index(drop=False)
            .rename(columns={"median_rpm_reverted": "total_reads"})
        )
    return achilles_pdna_count_totals


def _build_score_pdna_totals_table(score_pdna_df_path: Path) -> pd.DataFrame:
    with dask_client() as _:
        score_pdna = dd.read_csv(score_pdna_df_path, low_memory=False)
        score_pdna_count_totals = (
            score_pdna[["p_dna_batch", "read_counts"]]
            .groupby("p_dna_batch")["read_counts"]
            .count()
            .reset_index(drop=False)
            .rename(columns={"read_counts": "total_reads"})
            .compute()
        )
    return score_pdna_count_totals


@app.command()
def build_read_count_total_tables(
    depmap_modeling_df_path: Path,
    achilles_pdna_df_path: Path,
    score_pdna_df_path: Path,
    final_counts_table_out: Path,
    pdna_table_out: Path,
) -> None:
    _build_depmap_modeling_df_totals_table(depmap_modeling_df_path).to_csv(
        final_counts_table_out
    )
    pd.concat(
        [
            _build_achilles_pdna_totals_table(achilles_pdna_df_path),
            _build_score_pdna_totals_table(score_pdna_df_path),
        ]
    ).to_csv(pdna_table_out)
    return None


if __name__ == "__main__":
    app()
