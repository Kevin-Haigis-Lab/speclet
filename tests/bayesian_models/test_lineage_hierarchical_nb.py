import arviz as az
import pandas as pd
import pymc as pm
import pytest
from pymc.backends.base import MultiTrace

from speclet.bayesian_models.lineage_hierarchical_nb import (
    LineageHierNegBinomModel as LHNBModel,
)
from speclet.bayesian_models.lineage_hierarchical_nb import (
    _get_cancer_genes_accounting_for_sublineage,
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


def _set_kras_mutants(data: pd.DataFrame) -> pd.DataFrame:
    data = data.query("hugo_symbol != 'KRAS'").reset_index(drop=True)
    copy_gene = data["hugo_symbol"].unique()[0]
    cell_lines = data["depmap_id"].unique()
    muts = cell_lines[: (len(cell_lines) // 2)]

    kras_data = (
        data.copy()
        .query(f"hugo_symbol == '{copy_gene}'")
        .reset_index(drop=True)
        .assign(hugo_symbol="KRAS", sgrna="kras-fake-sgrna", is_mutated=False)
    )
    kras_data.loc[kras_data["depmap_id"].isin(muts), "is_mutated"] = True
    return pd.concat([data, kras_data]).reset_index(drop=True)


@pytest.mark.parametrize(["min_frac", "does_have_vars"], [(1.0, False), (0.0, True)])
def test_var_h_only_when_cancer_genes(
    crc_data: pd.DataFrame,
    min_frac: float,
    does_have_vars: bool,
) -> None:
    crc_data = _set_kras_mutants(crc_data)
    model = LHNBModel(
        lineage="colorectal", min_n_cancer_genes=0, min_frac_cancer_genes=min_frac
    )
    model_data = model.make_data_structure(model.data_processing_pipeline(crc_data))
    if min_frac > 0:
        assert model_data.CG == 0
    else:
        assert model_data.CG > 0

    pm_model = model.pymc_model(crc_data)
    var_names = [v.name for v in pm_model.unobserved_RVs]
    print(var_names)
    if does_have_vars:
        assert "h" in var_names
        assert "sigma_h" in var_names
    else:
        assert "h" not in var_names
        assert "sigma_h" not in var_names
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


def test_specific_colorectal_cancer_genes_used() -> None:
    cancer_genes = _get_cancer_genes_accounting_for_sublineage("colorectal")
    assert cancer_genes == {"KRAS", "APC", "FBXW7", "PIK3CA"}
