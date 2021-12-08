"""Check a dictionary of keyword arguments against the parameters of a function."""

import inspect
import warnings
from typing import Callable, Mapping, Optional


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


class KeywordsWillBePassedToKwargsWarning(UserWarning):
    """Keywords will be passed to 'kwargs'."""

    pass


def _get_function_parameters(f: Callable) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(f).parameters


def _look_for_kwargs_parameters(params: Mapping[str, inspect.Parameter]) -> bool:
    for param in params.values():
        if param.kind is inspect._ParameterKind.VAR_KEYWORD:
            return True
    return False


def check_kwarg_dict(
    keywords: list[str],
    f: Callable,
    blacklist: Optional[set[str]] = None,
    ignore_kwargs: bool = False,
) -> None:
    """Check a list of keywords against the parameters in a callable (e.g. function).

    Args:
        keywords (list[str]): List of expected keywords.
        f (Callable): Callable object that will be passed they keyword arguments.
        blacklist (tuple[str]): Parameters to ignore (e.g. `("self",)`)
        ignore_kwargs (bool, optional): Ignore keyword argument parameters? Defaults to
          False.

    Raises:
        KeywordsNotInCallableParametersError: Raised if spare keywords are found.

    Returns:
        None: None
    """
    parameters = _get_function_parameters(f)
    if ignore_kwargs:
        parameters = {
            name: param
            for name, param in parameters.items()
            if param.kind is not inspect._ParameterKind.VAR_KEYWORD
        }
    param_names: list[str] = list(parameters)
    is_kwargs = _look_for_kwargs_parameters(parameters)

    if blacklist is not None and len(blacklist) > 0:
        param_names = [n for n in param_names if n not in blacklist]

    spare_keys = [k for k in keywords if k not in param_names]

    if len(spare_keys) and not is_kwargs:
        raise KeywordsNotInCallableParametersError(spare_keys)
    elif len(spare_keys) and is_kwargs:
        warnings.warn(
            f"The following keys will be passed to kwargs: {spare_keys}",
            KeywordsWillBePassedToKwargsWarning,
        )
    return None
