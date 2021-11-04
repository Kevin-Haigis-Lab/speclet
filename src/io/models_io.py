#!/usr/bin/env python3

"""IO for Speclet models."""

from pathlib import Path

from src.io.general_io import project_root_dir


def models_dir() -> Path:
    """Path to the models directory.

    Returns:
        Path: Path to the models directory.
    """
    return project_root_dir() / "models"
