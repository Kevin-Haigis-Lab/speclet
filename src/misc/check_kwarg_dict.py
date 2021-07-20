"""Check a dictionary of keyword arguments against the parameters of a function."""

import inspect
from typing import Callable


class KeywordsNotInCallableParametersError(BaseException):
    """Keywords are not in a callable's available parameters."""

    def __init__(self, extra_keys: list[str]) -> None:
        """Create a error.

        Args:
            extra_keys (list[str]): Extra keys found.
        """
        self.extra_keys = extra_keys
        self.message = f"Spare keys: {self.extra_keys}"
        super().__init__(self.message)


def _get_parameter_names(f: Callable) -> list[str]:
    return list(inspect.signature(f).parameters)


def check_kwarg_dict(
    keywords: list[str], f: Callable, blacklist: tuple[str, ...] = ()
) -> None:
    """Check a list of keywords against the parameters in a callable (e.g. function).

    Args:
        keywords (list[str]): List of expected keywords.
        f (Callable): Callable object that will be passed they keyword arguments.
        blacklist (tuple[str]): Parameters to ignore (e.g. `("self",)`)

    Raises:
        KeywordsNotInCallableParametersError: Raised if spare keywords are found.

    Returns:
        None: None
    """
    param_names = _get_parameter_names(f)
    if len(blacklist) > 0:
        param_names = list(filter(lambda x: x not in blacklist, param_names))
    spare_keys = [k for k in keywords if k not in param_names]
    if len(spare_keys):
        raise KeywordsNotInCallableParametersError(spare_keys)
    return None
