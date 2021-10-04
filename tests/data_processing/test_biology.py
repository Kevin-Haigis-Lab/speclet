import pandas as pd
import pytest

from src.data_processing import biology as biohelp


@pytest.mark.parametrize(
    "input, chr, pos, strand",
    [
        ("chr2_130522105_-", "2", 130522105, -1),
        ("chr21_15_-", "21", 15, -1),
        ("chrM_576_+", "M", 576, +1),
        ("chrX_1_+", "X", 1, +1),
        ("chrY_8342790_-", "Y", 8342790, -1),
    ],
)
def test_extract_chromosome_location(
    input: str, chr: str, pos: int, strand: int
) -> None:
    input_df = pd.DataFrame({"my_col": pd.Series([input])})
    res = biohelp.extract_chromosome_location_to_df(input_df, col_name="my_col")
    print(res)
    assert res.shape == (1, 4)
    assert res.chr[0] == chr
    assert res.pos[0] == pos
    assert res.strand[0] == strand


def test_extract_chromosome_location_dataframe() -> None:
    input_df = pd.DataFrame(
        {
            "geno_info": [
                "chr2_130522105_-",
                "chr21_15_-",
                "chrM_576_+",
                "chrX_1_+",
                "chrY_8342790_-",
            ],
        }
    )
    output_df = pd.DataFrame(
        {
            "chr": ["2", "21", "M", "X", "Y"],
            "pos": [130522105, 15, 576, 1, 8342790],
            "strand": [-1, -1, 1, 1, -1],
        }
    )
    res = biohelp.extract_chromosome_location_to_df(input_df, col_name="geno_info")
    print(res)
    assert res.shape == (5, 4)
    assert all(res.chr == output_df.chr)
    assert all(res.pos == output_df.pos)
    assert all(res.strand == output_df.strand)
