from src.command_line_interfaces import cli_helpers


def test_clean_model_names():
    assert cli_helpers.clean_model_names("model_name") == "model_name"
    assert cli_helpers.clean_model_names("model name") == "model-name"
    assert cli_helpers.clean_model_names("model named Jerry") == "model-named-Jerry"
