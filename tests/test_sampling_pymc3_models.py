#!/usr/bin/env python3

from argparse import ArgumentParser as AP
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from typing import List

import numpy as np
import pandas as pd
import pretty_errors
import pytest

from analysis import sampling_pymc3_models as sampling
from analysis.common_data_processing import nunique, read_achilles_data

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


@pytest.fixture(scope="module")
def mock_achilles_data():
    return read_achilles_data(Path("tests", "depmap_test_data.csv"))


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


def test_common_idx_key_names(mock_achilles_data: pd.DataFrame):
    indices = sampling.common_indices(mock_achilles_data.sample(frac=1.0))
    for k in indices.keys():
        assert "idx" in k or "map" in k


def test_common_idx_sgrna_to_gene_map(mock_achilles_data: pd.DataFrame):
    indices = sampling.common_indices(mock_achilles_data.sample(frac=1.0))
    for sgrna in mock_achilles_data.sgrna.values:
        assert sgrna in indices["sgrna_to_gene_map"].sgrna.values
    for gene in mock_achilles_data.hugo_symbol.values:
        assert gene in indices["sgrna_to_gene_map"].hugo_symbol.values


def test_common_idx_depmap(mock_achilles_data: pd.DataFrame):
    indices = sampling.common_indices(mock_achilles_data.sample(frac=1.0))
    assert nunique(mock_achilles_data.depmap_id.values) == nunique(
        indices["cellline_idx"]
    )


def test_common_idx_pdna_batch(mock_achilles_data: pd.DataFrame):
    indices = sampling.common_indices(mock_achilles_data.sample(frac=1.0))
    assert nunique(mock_achilles_data.pdna_batch.values) == nunique(
        indices["batch_idx"]
    )


class TestCLI:
    @pytest.fixture
    def parser(self) -> AP:
        return AP(conflict_handler="resolve", exit_on_error=False, allow_abbrev=False)

    @pytest.fixture
    def default_args(self) -> List[str]:
        return ["--model", "crc-m1", "--name", "mock_model"]

    def test_get_model_and_name(self, parser: AP, default_args: List[str]):
        args = sampling.parse_cli_arguments(parser, default_args)
        assert args.name == "mock_model"
        assert args.model == "crc-m1"

    def test_error_model_not_available(self, parser: AP):
        with pytest.raises(Exception):
            _ = sampling.parse_cli_arguments(
                parser, ["--model", "not-real-model", "--name", "mock_model"]
            )

    def test_error_no_model(self, parser: AP):
        with pytest.raises(SystemExit):
            _ = sampling.parse_cli_arguments(parser, ["--name", "mock_model"])

    def test_error_no_name(self, parser: AP):
        with pytest.raises(SystemExit):
            _ = sampling.parse_cli_arguments(parser, ["--model", "crc-m1"])

    def test_force_sample(self, parser: AP, default_args: List[str]):
        args = sampling.parse_cli_arguments(parser, default_args)
        assert args.force_sample is False
        args = sampling.parse_cli_arguments(parser, default_args + ["--force-sample"])
        assert args.force_sample is True

    def test_debug(self, parser: AP, default_args: List[str]):
        args = sampling.parse_cli_arguments(parser, default_args)
        assert args.debug is False
        args = sampling.parse_cli_arguments(parser, default_args + ["--debug"])
        assert args.debug is True
        args = sampling.parse_cli_arguments(parser, default_args + ["-d"])
        assert args.debug is True

    def test_touch(self, parser: AP, default_args: List[str]):
        args = sampling.parse_cli_arguments(parser, default_args)
        assert args.touch is False
        args = sampling.parse_cli_arguments(parser, default_args + ["--touch"])
        assert args.touch is True

    def test_seed(self, parser: AP, default_args: List[str]):
        args = sampling.parse_cli_arguments(parser, default_args)
        assert args.random_seed is None
        args = sampling.parse_cli_arguments(parser, default_args + ["--random-seed"])
        assert args.random_seed is None
        args = sampling.parse_cli_arguments(
            parser, default_args + ["--random-seed", "123"]
        )
        assert args.random_seed == 123
        args = sampling.parse_cli_arguments(parser, default_args + ["-s", "123"])
        assert args.random_seed == 123
