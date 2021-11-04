"""SpecletNine model."""


from pathlib import Path
from typing import Final, Optional

import janitor  # noqa: F401
import pymc3 as pm
from pydantic import BaseModel

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import (
    Data,
    DataFrameTransformation,
    make_count_model_data_manager,
)
from src.models.speclet_model import (
    ObservedVarName,
    SpecletModel,
    SpecletModelDataManager,
)


class SpecletNineConfiguration(BaseModel):
    """Configuration for SpecletNine."""

    broad_only: bool = True


def _reduce_num_genes_for_dev(df: Data) -> Data:
    logger.warn("Reducing number of genes for development.")
    _genes = ["KRAS", "TP53", "NLRP8", "KLF5"]
    return df[df.hugo_symbol.isin(_genes)]


def _thin_data_columns(df: Data) -> Data:
    keep_cols: Final[list[str]] = [
        "sgrna",
        "hugo_symbol",
        "depmap_id",
        "lineage",
        "counts_initial_adj",
        "counts_final",
        "p_dna_batch",
        "screen",
    ]
    return df[keep_cols]


class SpecletNine(SpecletModel):
    """## SpecletNine.

    A negative binomial model of the read counts from the CRISPR screen data.
    """

    _config: SpecletNineConfiguration

    def __init__(
        self,
        name: str,
        data_manager: Optional[SpecletModelDataManager] = None,
        root_cache_dir: Optional[Path] = None,
        config: Optional[SpecletNineConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletNine model.

        Args:
            name (str): A unique identifier for this instance of SpecletNine. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletNineConfiguration, optional): Model configuration.
        """
        logger.info("Creating a new SpecletNine object.")
        self._config = SpecletNineConfiguration() if config is None else config

        if data_manager is None:
            _other_transforms: list[DataFrameTransformation] = []
            _other_transforms.append(_thin_data_columns)
            if self._config.broad_only:
                _other_transforms.append(achelp.filter_for_broad_source_only)

            data_manager = make_count_model_data_manager(
                DataFile.DEPMAP_CRC_BONE_SUBSAMPLE, other_transforms=_other_transforms
            )

        super().__init__(name, data_manager, root_cache_dir=root_cache_dir)

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Define the PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
            ObservedVarName: Name of the target variable in the model.
        """
        logger.info("Creating SpecletNine model.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        logger.info(f"Number of data points: {total_size}")
        co_idx = achelp.common_indices(data)
        logger.info(f"Number of sgRNA: {co_idx.n_sgrnas}")
        logger.info(f"Number of genes: {co_idx.n_genes}")
        logger.info(f"Number of cell lines: {co_idx.n_celllines}")
        logger.info(f"Number of lineages: {co_idx.n_lineages}")
        # b_idx = achelp.data_batch_indices(data)

        logger.info("Creating coordinates dictionary.")
        coords = {
            "one": ["dim_one"],
            "sgrna": data.sgrna.cat.categories,
            "gene": data.hugo_symbol.cat.categories,
            "cell_line": data.depmap_id.cat.categories,
            # "lineage": data.lineage.cat.categories,
        }

        logger.info("Building PyMC3 model.")
        with pm.Model(coords=coords) as model:
            s = pm.Data("sgrna_idx", co_idx.sgrna_idx)
            g_s = pm.Data("sgrna_to_gene_idx", co_idx.sgrna_to_gene_idx)
            c = pm.Data("cell_line_idx", co_idx.cellline_idx)
            # l_c = pm.Data("cell_line_to_lineage_idx", co_idx.cellline_to_lineage_idx)
            ct_initial = pm.Data("ct_initial", data.counts_initial_adj.values)
            ct_final = pm.Data("ct_final", data.counts_final.values)

            # shape: [gene x cell line]
            mu_beta = pm.Normal("mu_beta", 0, 0.1, dims=("gene", "one"))
            sigma_beta = pm.Exponential("sigma_beta", 2)

            beta = pm.Normal(
                "beta",
                mu_beta[g_s, :],
                sigma_beta,
                dims=("sgrna", "cell_line"),
            )

            eta = pm.Deterministic("eta", beta[s, c])
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            alpha = pm.Exponential("alpha", 1)
            y = pm.NegativeBinomial(  # noqa: F841
                "y", ct_initial * mu, alpha, observed=ct_final
            )

        return model, "y"
