from pathlib import Path

import pytest


@pytest.fixture
def mock_model_config() -> Path:
    return Path("tests/models/mock-model-config.yaml")
