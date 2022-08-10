import arviz as az
import pandas as pd
import pymc as pm
import pytest
from pymc.backends.base import MultiTrace

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


@pytest.mark.slow
def test_chol_cov_coords_pmsample(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    trace: az.InferenceData | MultiTrace | None = None
    with crc_lhnb_model.pymc_model(crc_data):
        trace = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            idata_kwargs={"dims": crc_lhnb_model.additional_variable_dims()},
        )

    assert isinstance(trace, az.InferenceData)
    posterior = trace.get("posterior")
    assert posterior is not None
    # Genes
    assert posterior["genes_chol_cov_stds"].dims[2] == "genes_params"
    assert posterior["genes_chol_cov_corr"].dims[2] == "genes_params"
    assert posterior["genes_chol_cov_corr"].dims[3] == "genes_params_"
    # Cell lines
    assert posterior["cells_chol_cov_stds"].dims[2] == "cells_params"
    assert posterior["cells_chol_cov_corr"].dims[2] == "cells_params"
    assert posterior["cells_chol_cov_corr"].dims[3] == "cells_params_"
    # Some other variables (skipped - bug in PyMC, submitted PR)
    # for v in ["b", "d"]:
    #     assert posterior[v].dims[3] == "gene"
    # for v in ["mu_k", "mu_m"]:
    #     assert posterior[v].dims[3] == "cell_line"
    return None


def test_posterior_checks_use_available_variable_names(
    crc_data: pd.DataFrame, crc_lhnb_model: LHNBModel
) -> None:
    model = crc_lhnb_model.pymc_model(crc_data)
    _n_checks = 0
    for check in crc_lhnb_model.posterior_sample_checks():
        if (v := getattr(check, "var_name")) is not None:
            _n_checks += 1
            assert getattr(model, v) is not None
    assert _n_checks > 1
    return None
