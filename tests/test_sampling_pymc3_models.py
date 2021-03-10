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


#### ---- CLI ---- ####


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
