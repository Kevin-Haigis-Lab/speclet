import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from src.data_processing import vectors as vhelp


@st.composite
def my_arrays(draw, min_size: int = 1):
    return draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=100.0,
                allow_infinity=False,
                allow_nan=False,
            ),
            min_size=min_size,
        )
    )


@given(
    st.floats(allow_nan=False), st.floats(allow_nan=False), st.floats(allow_nan=False)
)
def test_squish(val: float, lower_bound: float, upper_bound: float):
    assume(lower_bound < upper_bound)
    squished_val = vhelp.squish(val, lower=lower_bound, upper=upper_bound)
    assert lower_bound <= squished_val <= upper_bound


@given(my_arrays(), st.floats(allow_nan=False), st.floats(allow_nan=False))
def test_squish_array(ary: np.ndarray, lower_bound: float, upper_bound: float):
    assume(lower_bound < upper_bound)
    squished_ary = vhelp.squish_array(ary, lower=lower_bound, upper=upper_bound)
    assert np.all(lower_bound <= squished_ary)
    assert np.all(squished_ary <= upper_bound)


@given(my_arrays())
def test_careful_zscore(ary: np.ndarray):
    z_ary = vhelp.careful_zscore(ary)
    assert np.mean(z_ary) == pytest.approx(0.0, abs=0.01)
    assert np.allclose(z_ary, 0.0) or np.std(z_ary) == pytest.approx(1.0, abs=0.01)


@given(hnp.arrays(np.float32, shape=st.integers(0, 10)))
def test_np_identity_floats(ary: np.ndarray):
    i_ary = vhelp.np_identity(ary)
    np.testing.assert_array_equal(ary, i_ary)


@given(hnp.arrays(np.int64, shape=st.integers(0, 10)))
def test_np_identity_ints(ary: np.ndarray):
    i_ary = vhelp.np_identity(ary)
    np.testing.assert_array_equal(ary, i_ary)
