#!/usr/bin/env python3

"""Logger objects for use throughout the project."""


import logging
import pathlib
from typing import Union

from rich.logging import RichHandler


def _get_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.set_name("default-console-handler")
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("(%(levelname)s) %(message)s")
    handler.setFormatter(fmt)
    return handler


def _get_rich_console_handler() -> RichHandler:
    handler = RichHandler(level=logging.INFO)
    handler.set_name("default-rich-console-handler")
    return handler


def _get_log_file() -> pathlib.Path:
    dir = pathlib.Path(__file__).parent.parent.absolute() / "logs"
    if not dir.exists():
        dir.mkdir()
    return dir / "speclet.log"


def _get_file_handler() -> logging.Handler:
    handler = logging.FileHandler(_get_log_file())
    handler.set_name("default-file-handler")
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


def set_console_handler_level(to: Union[int, str]) -> None:
    """Set the consle handler level.

    Args:
        to (Union[int, str]): New log level for console handlers.

    Returns:
        None: None
    """
    for handler in logger.handlers:
        if (handler_name := handler.name) is not None:
            if "console" in handler_name:
                handler.setLevel(to)
