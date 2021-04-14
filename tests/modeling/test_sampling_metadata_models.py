#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict

import pytest

from src.modeling.sampling_metadata_models import SamplingArguments

#### ---- SamplingArguments class ---- ####


class TestSamplingArguments:
    @pytest.fixture
    def mock_info(self) -> Dict[str, Any]:
        return {
            "name": "model name",
            "sample": False,
            "ignore_cache": True,
            "cache_dir": Path("fake_path/to_nowhere"),
            "debug": False,
            "random_seed": 123,
        }

    def test_manual_creation(self, mock_info: Dict[str, Any]):
        args = SamplingArguments(
            name=mock_info["name"],
            sample=mock_info["sample"],
            ignore_cache=mock_info["ignore_cache"],
            debug=mock_info["debug"],
            random_seed=mock_info["random_seed"],
        )
        assert isinstance(args, SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.sample == mock_info["sample"]
        assert args.ignore_cache == mock_info["ignore_cache"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_creation_from_dict(self, mock_info: Dict[str, Any]):
        args = SamplingArguments(**mock_info)
        assert isinstance(args, SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.sample == mock_info["sample"]
        assert args.ignore_cache == mock_info["ignore_cache"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_optional_params(self, mock_info: Dict[str, Any]):
        _ = mock_info.pop("random_seed", None)
        args = SamplingArguments(**mock_info)
        assert args.random_seed is None

    def test_extra_keyvalues_in_dict(self, mock_info: Dict[str, Any]):
        mock_info["A"] = "B"
        mock_info["123"] = "fieorj"
        mock_info["0"] = 980
        mock_info["12.34"] = ["some", "list"]

        args = SamplingArguments(**mock_info)

        assert isinstance(args, SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.sample == mock_info["sample"]
        assert args.ignore_cache == mock_info["ignore_cache"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_missing_keys_raise_error(self, mock_info: Dict[str, Any]):
        _ = mock_info.pop("name", None)
        with pytest.raises(Exception):
            _ = SamplingArguments(**mock_info)

    def test_keys_must_be_strings(self, mock_info: Dict[Any, Any]):
        mock_info[123] = "ABC"
        with pytest.raises(TypeError):
            _ = SamplingArguments(**mock_info)
