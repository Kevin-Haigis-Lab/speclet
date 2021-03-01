#!/usr/bin/env python3

from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from typing import List

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pytest

from analysis import sampling_pymc3_models as sampling
from analysis.common_data_processing import make_cat

#### ---- User Messages ---- ####


def test_print_model(capsys: pytest.CaptureFixture):
    name = "SOME MODEL NAME"
    sampling.print_model(name)
    captured = capsys.readouterr()
    assert name in captured.out


def test_print_info(capsys: pytest.CaptureFixture):
    msg = "here is some info"
    sampling.info(msg)
    captured = capsys.readouterr()
    assert msg in captured.out


def test_print_done(capsys: pytest.CaptureFixture):
    sampling.done()
    captured = capsys.readouterr()
    assert "Done" in captured.out


#### ---- File IO ---- ####


def test_make_cache_name():
    name = "MOCK_MOCK_NAME"
    p = sampling.make_cache_name(name)
    assert isinstance(p, Path)
    assert p.name == name


def test_loading_of_crc_data():
    df = sampling.load_crc_data(debug=True)
    assert isinstance(df, pd.DataFrame)
    for col in ["sgrna", "hugo_symbol", "depmap_id", "gene_cn"]:
        assert col in df.columns


#### ---- Misc. ---- ####


def test_crc_batch_sizes_are_logical():
    assert sampling.crc_batch_size(debug=True) < sampling.crc_batch_size(debug=False)


#### ---- Index helpers ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(ascii_lowercase), of_length, replace=True))


def test_generation_of_mock_sgrna():
    for i in np.random.randint(1, 100, 100):
        sgrna = make_mock_sgrna(i)
        assert isinstance(sgrna, str)
        assert len(sgrna) == i


@pytest.fixture
def mock_gene_data() -> pd.DataFrame:
    genes = list(ascii_uppercase[:5])
    gene_list: List[str] = []
    sgrna_list: List[str] = []
    for gene in genes:
        for _ in range(np.random.randint(5, 10)):
            gene_list.append(gene)
            sgrna_list.append(make_mock_sgrna(20))

    df = pd.DataFrame({"hugo_symbol": gene_list, "sgrna": sgrna_list}, dtype="category")
    df = pd.concat([df] + [df.sample(frac=0.75) for _ in range(10)])
    df = df.sample(frac=1.0)
    df["y"] = np.random.randn(len(df))
    return df


def test_sgrna_to_gene_mapping_df_is_smaller(mock_gene_data: pd.DataFrame):
    sgrna_map = sampling.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map) < len(mock_gene_data)
    assert sgrna_map["hugo_symbol"].dtype == "category"


def test_sgrna_to_gene_map_preserves_categories(mock_gene_data: pd.DataFrame):
    sgrna_map = sampling.make_sgrna_to_gene_mapping_df(mock_gene_data)
    for col in sgrna_map.columns:
        assert sgrna_map[col].dtype == "category"


def test_sgrna_are_unique(mock_gene_data: pd.DataFrame):
    sgrna_map = sampling.make_sgrna_to_gene_mapping_df(mock_gene_data)
    assert len(sgrna_map["sgrna"].values) == len(sgrna_map["sgrna"].values.unique())


def test_different_colnames(mock_gene_data: pd.DataFrame):
    df = mock_gene_data.rename(columns={"sgrna": "a", "hugo_symbol": "b"})
    sgrna_map_original = sampling.make_sgrna_to_gene_mapping_df(mock_gene_data)
    sgrna_map_new = sampling.make_sgrna_to_gene_mapping_df(
        df, sgrna_col="a", gene_col="b"
    )
    for col in ["a", "b"]:
        assert col in sgrna_map_new.columns
    for col_i in range(sgrna_map_new.shape[1]):
        np.testing.assert_array_equal(
            sgrna_map_new.iloc[:, col_i].values,
            sgrna_map_original.iloc[:, col_i].values,
        )
