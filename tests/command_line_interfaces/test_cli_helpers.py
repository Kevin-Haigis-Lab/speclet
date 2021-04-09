#!/usr/bin/env python3


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
