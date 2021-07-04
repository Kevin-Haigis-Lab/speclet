#!/usr/bin/env python3

"""Helper functions for the various CLIs."""


import pretty_errors

#### ---- Pretty Errors ---- ####


def configure_pretty() -> None:
    """Configure 'pretty' for better Python error messages."""
    pretty_errors.configure(
        filename_color=pretty_errors.BLUE,
        code_color=pretty_errors.BLACK,
        exception_color=pretty_errors.BRIGHT_RED,
        exception_arg_color=pretty_errors.RED,
        line_color=pretty_errors.BRIGHT_BLACK,
    )


#### ---- Models ---- ####


def clean_model_names(n: str) -> str:
    """Clean a custom model name.

    Args:
        n (str): Custom model name.

    Returns:
        str: Cleaned model name.
    """
    return n.replace(" ", "-")
