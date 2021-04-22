#!/usr/bin/env python3

from itertools import product
from pathlib import Path
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.data_processing import common as dphelp
from src.data_processing.achilles import zscale_cna_by_group
from src.managers.model_data_managers import CrcDataManager

#### ---- Mock data ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(letters), of_length, replace=True))


@pytest.fixture
def mock_data() -> pd.DataFrame:
    genes = np.random.choice(list(LETTERS), 10, replace=False)
    sgrna_to_gene_map: Dict[str, str] = {}
    for gene in genes:
        for _ in range(np.random.randint(3, 6)):
            sgrna_to_gene_map[make_mock_sgrna()] = gene

    cell_lines = ["line" + str(i) for i in range(5)]
    pdna_batches = ["batch" + str(i) for i in range(3)]
    df = pd.DataFrame(
        product(sgrna_to_gene_map.keys(), cell_lines), columns=["sgrna", "depmap_id"]
    )
    df["hugo_symbol"] = [sgrna_to_gene_map[s] for s in df.sgrna.values]
    df["pdna_batch"] = np.random.choice(pdna_batches, len(df), replace=True)

    df.sort_values(["hugo_symbol", "sgrna"])
    for col in df.columns:
        df = dphelp.make_cat(df, col)

    df["gene_cn"] = np.abs(np.random.normal(2, 0.1, len(df)))
    df["log2_cn"] = np.log2(df.gene_cn + 1)
    df = zscale_cna_by_group(
        df,
        gene_cn_col="log2_cn",
        new_col="z_log2_cn",
        groupby_cols=["depmap_id"],
        cn_max=np.log2(10),
    )
    df["lfc"] = np.random.randn(len(df))
    return df


#### ---- Test CrcDataManager ---- ####


class TestCrcDataManager:
    def test_batch_size(self, tmp_path: Path):
        model = CrcDataManager()
        not_debug_batch_size = model.get_batch_size()
        model.debug = True
        debug_batch_size = model.get_batch_size()
        assert debug_batch_size < not_debug_batch_size

    def test_data_paths(self, tmp_path: Path):
        model = CrcDataManager(debug=True)
        assert model.get_data_path().exists and model.get_data_path().is_file()

    def test_get_data(self, tmp_path: Path):
        model = CrcDataManager(debug=True)
        assert model.data is None
        data = model.get_data()
        assert model.data is not None
        assert model.data.shape[0] > model.data.shape[1]
        assert model.data.shape[0] == data.shape[0]
        assert model.data.shape[1] == data.shape[1]
