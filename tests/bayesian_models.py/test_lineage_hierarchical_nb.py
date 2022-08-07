import pandas as pd
import pytest

from speclet.bayesian_models.lineage_hierarchical_nb import (
    LineageHierNegBinomModel as LHNBModel,
)


@pytest.fixture
def crc_data(depmap_test_df: pd.DataFrame) -> pd.DataFrame:
    depmap_test_df = depmap_test_df.copy()
    depmap_test_df["lineage"] = ["colorectal"] * len(depmap_test_df)
    return depmap_test_df


@pytest.fixture
def crc_lhnb_model() -> LHNBModel:
    return LHNBModel(lineage="colorectal")


def test_raise_error_multiple_lineages(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    crc_data["lineage"] = (["colorectal", "bone"] * len(crc_data))[: len(crc_data)]
    assert crc_data["lineage"].nunique() == 2
    with pytest.raises(BaseException):
        _ = crc_lhnb_model.pymc_model(crc_data)
    return None


def test_raise_error_wrong_lineage(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    crc_data["lineage"] = ["bone"] * len(crc_data)
    with pytest.raises(BaseException):
        _ = crc_lhnb_model.pymc_model(crc_data)
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
