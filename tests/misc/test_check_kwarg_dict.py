from typing import Any, Callable

import pytest

from src.misc import check_kwarg_dict


def f1() -> None:
    return None


def f2(arg1: str) -> int:
    return 2


def f3(*args: Any, **kwargs: Any) -> int:
    return 2


mock_functions = [f1, f2, f3]
mock_function_args = [[], ["arg1"], ["args", "kwargs"]]


@pytest.mark.parametrize("fxn", [f1, f2])
def test_check_kwarg_dict_with_functions_raises(fxn: Callable) -> None:
    with pytest.raises(check_kwarg_dict.KeywordsNotInCallableParametersError):
        check_kwarg_dict.check_kwarg_dict(["not_a_param"], fxn)


def test_check_kwarg_dict_with_functions_warns() -> None:
    with pytest.warns(check_kwarg_dict.KeywordsWillBePassedToKwargsWarning):
        check_kwarg_dict.check_kwarg_dict(["not_a_param"], f3)


@pytest.mark.parametrize("fxn, args", zip(mock_functions, mock_function_args))
def test_check_kwarg_dict_with_functions(fxn: Callable, args: list[str]) -> None:
    assert check_kwarg_dict.check_kwarg_dict(args, fxn) is None


class MyClass1:
    def __init__(self) -> None:
        return None

    def m(self) -> None:
        return None


class MyClass2:
    def __init__(self, arg: list[str]) -> None:
        return None

    def m(self, arg: Any) -> None:
        return None


class MyClass3:
    def __init__(self, arg1: str, arg2: int) -> None:
        return None

    def m(self, *args: Any, **kwargs: Any) -> None:
        return None


mock_classes = [MyClass1, MyClass2, MyClass3]
mock_class_init_args = [[], ["arg"], ["arg1", "arg2"]]
mock_class_method_args = [[], ["arg"], ["args", "kwargs"]]


@pytest.mark.parametrize("cls", mock_classes)
def test_check_kwarg_dict_with_class_init_raises(cls: Callable) -> None:
    with pytest.raises(check_kwarg_dict.KeywordsNotInCallableParametersError):
        check_kwarg_dict.check_kwarg_dict(["not_a_param"], cls)


@pytest.mark.parametrize("cls, args", zip(mock_classes, mock_class_init_args))
def test_check_kwarg_dict_with_class_init(cls: Callable, args: list[str]) -> None:
    assert check_kwarg_dict.check_kwarg_dict(args, cls) is None


@pytest.mark.parametrize(
    "cls, args", [(MyClass2, ["arg"]), (MyClass3, ["arg1", "arg2"])]
)
def test_check_kwarg_dict_with_class_init_blacklist(
    cls: Callable, args: list[str]
) -> None:
    with pytest.raises(check_kwarg_dict.KeywordsNotInCallableParametersError):
        check_kwarg_dict.check_kwarg_dict(args, cls, blacklist=tuple(args))


@pytest.mark.parametrize("cls, args", zip(mock_classes, mock_class_method_args))
def test_check_kwarg_dict_with_class_method_raises(
    cls: Callable, args: list[str]
) -> None:
    assert check_kwarg_dict.check_kwarg_dict(args, cls.m) is None  # type: ignore
