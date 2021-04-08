#!/usr/bin/env python3

"""Custom context managers."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union

# source:
#   https://thomaseckert.dev/posts/changing-directory-with-a-python-context-manager


@contextmanager
def set_directory(path: Union[Path, str]):
    """Set the current working directory within the context.

    Args:
        path (Path): The path to the desired current working directory.

    Returns:
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
