import os
from contextlib import contextmanager
from pathlib import Path


# source: https://thomaseckert.dev/posts/changing-directory-with-a-python-context-manager
@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)
