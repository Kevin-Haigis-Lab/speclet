#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict

import pretty_errors
import pytest

from src.modeling import sampling_pymc3_models as sampling
from src.modeling.sampling_pymc3_models import SamplingArguments

#### ---- File IO ---- ####


def test_make_cache_name():
    name = "MOCK_MOCK_NAME"
    p = sampling.make_cache_name(name)
    assert isinstance(p, Path)
    assert p.name == name


def test_clean_model_names():
    assert sampling.clean_model_names("model_name") == "model_name"
    assert sampling.clean_model_names("model name") == "model-name"
    assert sampling.clean_model_names("model named Jerry") == "model-named-Jerry"


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
            cache_dir=mock_info["cache_dir"],
            debug=mock_info["debug"],
            random_seed=mock_info["random_seed"],
        )
        assert isinstance(args, SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.sample == mock_info["sample"]
        assert args.ignore_cache == mock_info["ignore_cache"]
        assert args.cache_dir == mock_info["cache_dir"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_creation_from_dict(self, mock_info: Dict[str, Any]):
        args = SamplingArguments(**mock_info)
        assert isinstance(args, SamplingArguments)
        assert args.name == mock_info["name"]
        assert args.sample == mock_info["sample"]
        assert args.ignore_cache == mock_info["ignore_cache"]
        assert args.cache_dir == mock_info["cache_dir"]
        assert args.debug == mock_info["debug"]
        assert args.random_seed == mock_info["random_seed"]

    def test_optional_params(self, mock_info: Dict[str, Any]):
        _ = mock_info.pop("random_seed", None)
        args = SamplingArguments(**mock_info)
        assert args.random_seed is None

    def test_string_to_path(self, mock_info: Dict[str, Any]):
        mock_info["cache_dir"] = "another_path/but/written/as/a/string"
        args = SamplingArguments(**mock_info)
        assert args.cache_dir == Path(mock_info["cache_dir"])

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
        assert args.cache_dir == mock_info["cache_dir"]
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
