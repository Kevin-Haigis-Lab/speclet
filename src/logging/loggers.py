#!/usr/bin/env python3

"""Logger objects for use throughout the project."""


import logging
import logging.config
import pathlib

_logging_conf_path = pathlib.Path(__file__).parent.absolute()

logging.config.fileConfig(fname=_logging_conf_path / "loggers.conf")
logger = logging.getLogger("specletLogger")
