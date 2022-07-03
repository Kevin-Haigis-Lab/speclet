import pandas as pd
import pytest

from speclet.bayesian_models.lineage_hierarchical_nb import (
    LineageHierNegBinomModel as LHNBModel,
)
from speclet.modeling.pymc_helpers import get_variable_names


@pytest.fixture
def crc_data(depmap_test_df: pd.DataFrame) -> pd.DataFrame:
    depmap_test_df = depmap_test_df.copy()
    depmap_test_df["lineage"] = ["colorectal"] * len(depmap_test_df)
    return depmap_test_df


@pytest.fixture
def crc_lhnb_model() -> LHNBModel:
    return LHNBModel(lineage="colorectal")


def test_adjusting_to_number_of_screens(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    crc_data["screen"] = ["broad"] * len(crc_data)
    screen_params = ["sigma_p", "delta_p", "p"]

    # Should drop the `p` variable.
    model = crc_lhnb_model.pymc_model(crc_data)
    for param in screen_params:
        assert param not in get_variable_names(model)

    # Should include the `p` variable.
    crc_data["screen"] = (["sanger", "broad"] * len(crc_data))[: len(crc_data)]
    model = LHNBModel(lineage="colorectal").pymc_model(crc_data)
    for param in screen_params:
        assert param in get_variable_names(model)

    return None


def test_raise_error_only_one_cell_line(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    crc_data["depmap_id"] = ["my-cell-line"] * len(crc_data)
    with pytest.raises(BaseException):
        _ = crc_lhnb_model.pymc_model(crc_data)
    return None


def test_raise_error_only_one_gene(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    crc_data["hugo_symbol"] = ["my-gene"] * len(crc_data)
    with pytest.raises(BaseException):
        _ = crc_lhnb_model.pymc_model(crc_data)
    return None
