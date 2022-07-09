#!/usr/bin/env python3

"""Logger objects for use throughout the project."""


import logging
import pathlib

from rich.logging import RichHandler


def _get_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("(%(levelname)s) %(message)s")
    handler.setFormatter(fmt)
    return handler


def _get_rich_console_handler() -> RichHandler:
    return RichHandler(level=logging.INFO)


def _get_log_file() -> pathlib.Path:
    dir = pathlib.Path(__file__).parent.parent.absolute() / "logs"
    if not dir.exists():
        dir.mkdir()
    return dir / "speclet.log"


def _get_file_handler() -> logging.Handler:
    handler = logging.FileHandler(_get_log_file())
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s "
        + "[(%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


logger = logging.getLogger("speclet")
logger.setLevel(logging.DEBUG)
if len(logger.handlers) == 0:
    # logger.addHandler(_get_console_handler())
    logger.addHandler(_get_rich_console_handler())
    logger.addHandler(_get_file_handler())
    # Update `_idx_console_loggers` in `set_console_handler_level()` if needed.


def set_console_handler_level(to: int | str) -> None:
    """Set the console handler level.

    Args:
        to (Union[int, str]): New log level for console handlers.

    Returns:
        None: None
    """
    _idx_console_loggers = [0]  #
    for idx in _idx_console_loggers:
        handler = logger.handlers[idx]
        handler.setLevel(to)
