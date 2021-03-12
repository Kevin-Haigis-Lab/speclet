#!/usr/bin/env python3

from argparse import ArgumentParser as AP
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pretty_errors
import pytest

from analysis import sampling_pymc3_models as sampling

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


#### ---- SamplingArguments class ---- ####


class TestSamplingArguments:
    @pytest.fixture
    def mock_info(self) -> Dict[str, Any]:
        return {
            "name": "model name",
            "force_sampling": True,
            "cache_dir": Path("fake_path/to_nowhere"),
            "debug": False,
            "random_seed": 123,
        }

    def test_manual_creation(self, mock_info: Dict[str, Any]):
        args = sampling.SamplingArguments(
            name=mock_info["name"],
            force_sampling=mock_info["force_sampling"],
            cache_dir=mock_info["cache_dir"],
            debug=mock_info["debug"],
            random_seed=mock_info["random_seed"],
        )
        assert isinstance(args, sampling.SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.force_sampling == mock_info["force_sampling"]
        assert args.cache_dir == mock_info["cache_dir"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_creation_from_dict(self, mock_info: Dict[str, Any]):
        args = sampling.SamplingArguments(**mock_info)
        assert isinstance(args, sampling.SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.force_sampling == mock_info["force_sampling"]
        assert args.cache_dir == mock_info["cache_dir"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_optional_params(self, mock_info: Dict[str, Any]):
        _ = mock_info.pop("random_seed", None)
        args = sampling.SamplingArguments(**mock_info)
        assert args.random_seed is None

    def test_string_to_path(self, mock_info: Dict[str, Any]):
        mock_info["cache_dir"] = "another_path/but/written/as/a/string"
        args = sampling.SamplingArguments(**mock_info)
        assert args.cache_dir == Path(mock_info["cache_dir"])

    def test_extra_keyvalues_in_dict(self, mock_info: Dict[str, Any]):
        mock_info["A"] = "B"
        mock_info["123"] = "fieorj"
        mock_info["0"] = 980
        mock_info["12.34"] = ["some", "list"]

        args = sampling.SamplingArguments(**mock_info)

        assert isinstance(args, sampling.SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.force_sampling == mock_info["force_sampling"]
        assert args.cache_dir == mock_info["cache_dir"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_missing_keys_raise_error(self, mock_info: Dict[str, Any]):
        _ = mock_info.pop("name", None)
        with pytest.raises(Exception):
            _ = sampling.SamplingArguments(**mock_info)

    def test_keys_must_be_strings(self, mock_info: Dict[Any, Any]):
        mock_info[123] = "ABC"
        with pytest.raises(TypeError):
            _ = sampling.SamplingArguments(**mock_info)
