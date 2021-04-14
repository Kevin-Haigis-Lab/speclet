#!/usr/bin/env python3

from pathlib import Path
from typing import List

import pytest

from src.command_line_interfaces import cli_helpers
from src.models.crc_ceres_mimic import CrcCeresMimic
from src.models.crc_model_one import CrcModelOne

#### ---- Models ---- ####


def test_clean_model_names():
    assert cli_helpers.clean_model_names("model_name") == "model_name"
    assert cli_helpers.clean_model_names("model name") == "model-name"
    assert cli_helpers.clean_model_names("model named Jerry") == "model-named-Jerry"


def test_get_model_class():
    m1 = cli_helpers.get_model_class(cli_helpers.ModelOption.crc_model_one)
    assert m1 == CrcModelOne

    m2 = cli_helpers.get_model_class(cli_helpers.ModelOption.crc_ceres_mimic)
    assert m2 == CrcCeresMimic


#### ---- Modifying models ---- ####


@pytest.fixture
def ceres_model(tmp_path: Path) -> CrcCeresMimic:
    return CrcCeresMimic(name="TEST-MODEL", root_cache_dir=Path(tmp_path), debug=True)


@pytest.fixture
def model_names() -> List[str]:
    return ["model", "ceres-model", "CERES-model", "pymc3-ceres", "pymc3 ceres"]


def test_modify_ceres_model_by_name_nochange(
    ceres_model: CrcCeresMimic, model_names: List[str]
):
    for name in model_names:
        cli_helpers.modify_ceres_model_by_name(ceres_model, name)
        assert not ceres_model.copynumber_cov
        assert not ceres_model.sgrna_intercept_cov


def test_modify_ceres_model_by_name_sgrna(
    ceres_model: CrcCeresMimic, model_names: List[str]
):
    model_names_a = [n + "_sgrnaint" for n in model_names]
    for name in model_names_a:
        cli_helpers.modify_ceres_model_by_name(ceres_model, name)
        assert not ceres_model.copynumber_cov
        assert ceres_model.sgrna_intercept_cov


def test_modify_ceres_model_by_name_copynumber(
    ceres_model: CrcCeresMimic, model_names: List[str]
):
    model_names_a = [n + "_copynumber" for n in model_names]
    for name in model_names_a:
        cli_helpers.modify_ceres_model_by_name(ceres_model, name)
        assert ceres_model.copynumber_cov
        assert not ceres_model.sgrna_intercept_cov


def test_modify_ceres_model_by_name_copynumber_sgrna(
    ceres_model: CrcCeresMimic, model_names: List[str]
):
    model_names_a = [n + "_copynumber-sgrnaint" for n in model_names]
    for name in model_names_a:
        cli_helpers.modify_ceres_model_by_name(ceres_model, name)
        assert ceres_model.copynumber_cov
        assert ceres_model.sgrna_intercept_cov
