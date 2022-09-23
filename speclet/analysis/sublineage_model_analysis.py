"""Some helpers for the sublineage model analysis."""

import re
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from speclet.bayesian_models import BayesianModel
from speclet.io import lineage_modeling_data_path, project_root
from speclet.managers.posterior_data_manager import PosteriorDataManagers as PostDataMen
from speclet.model_configuration import BayesianModelConfiguration as BayesModelConfig
from speclet.model_configuration import read_model_configurations
from speclet.project_configuration import (
    get_model_configuration_file,
    project_config_broad_only,
)
from speclet.project_enums import ModelFitMethod


def _default_config_path() -> Path:
    return project_root() / get_model_configuration_file()


def _get_lineage_subtype_model_configs(
    config_path: Path | None = None,
) -> list[BayesModelConfig]:
    if config_path is None:
        config_path = project_root() / get_model_configuration_file()
    model_configs = read_model_configurations(
        config_path, active_only=True
    ).configurations
    model_configs = [c for c in model_configs if "hnb-single-lineage" in c.name]
    model_configs = [
        c for c in model_configs if c.model is BayesianModel.LINEAGE_HIERARCHICAL_NB
    ]
    return model_configs


def load_sublineage_model_posteriors() -> PostDataMen:
    """Load sublineage model posterior data managers."""
    config_fp = _default_config_path()
    model_configs = _get_lineage_subtype_model_configs(config_fp)
    model_names = [c.name for c in model_configs]
    model_names.sort()
    pattern = r"(?<=hnb-single-lineage-).*$"
    sublineage_names = [re.findall(pattern, m)[0] for m in model_names]
    sublineage_names = [n.replace("_", " ") for n in sublineage_names]

    postmen = PostDataMen(
        names=model_names,
        fit_methods=ModelFitMethod.PYMC_NUMPYRO,
        config_paths=config_fp,
        keys=sublineage_names,
    )
    return postmen


sub_to_lineage_map = dict[str, str]
lineage_list = list[str]


def sublineage_to_lineage_map(
    postmen: PostDataMen,
) -> tuple[sub_to_lineage_map, lineage_list]:
    """Map of sublineages to lineages (and a list of lineages).

    Args:
        postmen (PostDataMen): Posterior data managers.

    Returns:
        tuple[sub_to_lineage_map, lineage_list]: Map of sublineages to lineages and a
        sorted list of lineages.
    """
    sub_to_lineage = {pm.id: pm.id.split(" (")[0] for pm in postmen.posteriors}
    lineages = list(set(sub_to_lineage.values()))
    lineages.sort()
    return sub_to_lineage, lineages


def _prostate_data_fp() -> Path:
    return lineage_modeling_data_path("prostate")


def get_sgrna_to_gene_map() -> pd.DataFrame:
    """Generate a map of sgRNA to the target gene.

    Uses the prostate data to get the mapping quickly.

    Returns:
        pd.DataFrame: Data frame with `sgrna` and `hugo_symbol` columns.
    """
    prostate_data = dd.read_csv(
        _prostate_data_fp(),
        low_memory=False,
        dtype={"age": "float64", "counts_final": "float64", "p_dna_batch": "object"},
    )
    if project_config_broad_only():
        prostate_data = prostate_data.query("screen == 'broad'")

    sgrna_data = (
        prostate_data[["sgrna", "hugo_symbol", "sgrna_target_chr", "sgrna_target_pos"]]
        .drop_duplicates()
        .compute()
        .sort_values(["hugo_symbol", "sgrna"])
        .reset_index(drop=True)
    )
    assert len(sgrna_data) == sgrna_data["sgrna"].nunique(), "sgRNA mapping not unique."
    return sgrna_data
