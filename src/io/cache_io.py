#!/usr/bin/env python3

"""Cache directory management for the speclet project."""

from pathlib import Path

from src.io.data_io import project_root_dir


def default_cache_dir() -> Path:
    """Get default cache directory.

    Returns:
        Path: Path to the default cache directory.
    """
    cache_dir: Path = project_root_dir() / "models"
    return cache_dir
