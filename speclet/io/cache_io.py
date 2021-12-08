#!/usr/bin/env python3

"""Cache directory management for the speclet project."""

from pathlib import Path

from speclet.io.data_io import project_root_dir


def default_cache_dir() -> Path:
    """Get default cache directory.

    Returns:
        Path: Path to the default cache directory.
    """
    return project_root_dir() / "models"


def temp_dir() -> Path:
    """Get default temporary directory.

    Returns:
        Path: Path to the default temporary directory.
    """
    return project_root_dir() / "temp"
