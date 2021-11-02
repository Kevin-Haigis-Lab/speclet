"""General IO needs."""

from pathlib import Path


def project_root_dir() -> Path:
    """Speclet root directory.

    Returns:
        Path: Path to root directory.
    """
    return Path(__file__).parent.parent.parent
