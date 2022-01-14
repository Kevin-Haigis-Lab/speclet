"""Dask utilities and helpers."""

from typing import Any

from dask.distributed import Client


def get_client(
    n_workers: int = 4,
    threads_per_worker: int = 4,
    memory_limit: str = "16GB",
    **kwargs: dict[str, Any],
) -> Client:
    """Create a Dask client.

    Args:
        n_workers (int, optional): Number of workers. Defaults to 4.
        threads_per_worker (int, optional): Threads per worker. Defaults to 4.
        memory_limit (str, optional): Memory per worker. Defaults to "16GB".
        kwargs (dict[str, Any], optional): Passed to Client.

    Yields:
        Client: Dask distributed computing client.
    """
    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        **kwargs,
    )
