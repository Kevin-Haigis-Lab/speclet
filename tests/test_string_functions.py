#!/bin/env python3

from string import ascii_lowercase

from analysis import string_functions

#### ---- str_replace ---- ####


def test_replacing_patterns_in_a_list_of_strings():
    s = ["hey", "there"]
    assert string_functions.str_replace(s, "e", "X") == ["hXy", "thXrX"]


def test_replacing_patterns_in_a_tuple_of_strings():
    s = ("hey", "there")
    assert string_functions.str_replace(s, "e", "X") == ["hXy", "thXrX"]


def test_replacing_patterns_in_a_string():
    s = "hey"
    assert string_functions.str_replace(s, "e", "X") == "hXy"


def test_replacing_patterns_in_a_list_of_strings_with_defaults():
    s = ["hey", "there"]
    assert string_functions.str_replace(s, "e") == ["h y", "th r "]


def test_replacing_patterns_in_a_string_with_defaults():
    s = "hey"
    assert string_functions.str_replace(s, "e") == "h y"


#### ---- prefixed_count ---- ####


def test_prefixed_count():
    assert string_functions.prefixed_count("#", 5) == ["#0", "#1", "#2", "#3", "#4"]


def test_prefixed_count_plus_1():
    assert string_functions.prefixed_count("#", 5, plus=1) == [
        "#1",
        "#2",
        "#3",
        "#4",
        "#5",
    ]


def test_prefixed_count_plus_1pt5():
    assert string_functions.prefixed_count("#", 5, plus=1.5) == [
        "#1.5",
        "#2.5",
        "#3.5",
        "#4.5",
        "#5.5",
    ]


#### ---- str_wrap ---- ####


def test_str_wrap():
    assert string_functions.str_wrap("a" * 20, 10) == "a" * 10 + "\n" + "a" * 10


def test_str_wrap_vectorized():
    strings = [letter * 20 for letter in ascii_lowercase]
    expected_output = [letter * 10 + "\n" + letter * 10 for letter in ascii_lowercase]
    assert string_functions.str_wrap(strings, 10) == expected_output
