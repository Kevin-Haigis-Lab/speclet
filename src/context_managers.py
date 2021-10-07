"""Custom context managers."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

from dask.distributed import Client

import src.utils.dask


@contextmanager
def set_directory(path: Union[Path, str]) -> Generator[None, None, None]:
    """Set the current working directory within the context.

    Source:
      https://thomaseckert.dev/posts/changing-directory-with-a-python-context-manager

    Args:
        path (Path): The path to the desired current working directory.

    Returns:
        Generator[None, None, None]: Nothing to return.
    """
    if isinstance(path, str):
        path = Path(path)

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


@contextmanager
def dask_client(
    n_workers: int = 4,
    threads_per_worker: int = 4,
    memory_limit: str = "16GB",
    **kwargs: dict[str, Any],
) -> Generator[Client, None, None]:
    """Create and close a Dask client.

    Args:
        n_workers (int, optional): Number of workers. Defaults to 4.
        threads_per_worker (int, optional): Threads per worker. Defaults to 4.
        memory_limit (str, optional): Memory per worker. Defaults to "16GB".
        kwargs (dict[str, Any], optional): Passed to Client.

    Yields:
        Generator[Client, None, None]: Client and some other generator stuff...
    """
    client = src.utils.dask.get_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        **kwargs,
    )
    try:
        yield client
    finally:
        client.close()
