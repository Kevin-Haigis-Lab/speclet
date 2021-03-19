#!/usr/bin/env python3

"""Logger objects for use throughout the project."""


import logging
import pathlib


def _get_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("(%(levelname)s) %(message)s")
    handler.setFormatter(fmt)
    return handler


def _get_log_file() -> pathlib.Path:
    dir = pathlib.Path(__file__).parent.parent.absolute() / "logs"
    if not dir.exists():
        dir.mkdir()
    return dir / "speclet.log"


def _get_file_handler() -> logging.Handler:
    handler = logging.FileHandler(_get_log_file())
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(filename)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


def get_logger() -> logging.Logger:
    """Get the speclet-wide logger.

    Returns:
        logging.Logger: Speclet-wide logger object.
    """
    logger = logging.getLogger("speclet")
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        logger.addHandler(_get_console_handler())
        logger.addHandler(_get_file_handler())
    return logger
