import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union


# source: https://thomaseckert.dev/posts/changing-directory-with-a-python-context-manager
@contextmanager
def set_directory(path: Union[Path, str]):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    if isinstance(path, str):
        path = Path(path)

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)
