from string import ascii_letters
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.modeling import feature_engineering as feng

chars = [str(i) for i in range(10)] + list(ascii_letters)


@st.composite
def copynumber_dataframe(draw: Callable, group_name: str) -> pd.DataFrame:
    groups = draw(st.lists(st.text(alphabet=chars), min_size=1))
    values = [
        draw(
            st.lists(
                st.floats(
                    min_value=-1000.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
            )
        )
        for _ in groups
    ]
    return (
        pd.DataFrame({group_name: groups, "copy_number": values})
        .explode("copy_number")
        .astype({"copy_number": float})
        .reset_index(drop=True)
    )


@given(copynumber_dataframe(group_name="depmap_id"))
def test_centered_copynumber_by_cellline(df: pd.DataFrame) -> None:
    mod_df = feng.centered_copynumber_by_cellline(df.copy())
    for cell_line in mod_df["depmap_id"].unique():
        avg = mod_df.query(f"depmap_id == '{cell_line}'")["copy_number_cellline"].mean()
        assert avg == pytest.approx(0.0, abs=0.001)


@given(copynumber_dataframe(group_name="hugo_symbol"))
def test_centered_copynumber_by_gene(df: pd.DataFrame) -> None:
    mod_df = feng.centered_copynumber_by_gene(df.copy())
    for gene in mod_df["hugo_symbol"].unique():
        avg = mod_df.query(f"hugo_symbol == '{gene}'")["copy_number_gene"].mean()
        assert avg == pytest.approx(0.0, abs=0.001)


@given(st.lists(st.booleans(), max_size=100))
def test_converting_is_mutated_column(is_mutated: list[bool]) -> None:
    df = pd.DataFrame({"is_mutated": is_mutated})
    mod_df = feng.convert_is_mutated_to_numeric(df)
    assert mod_df["is_mutated"].dtype == np.int64
