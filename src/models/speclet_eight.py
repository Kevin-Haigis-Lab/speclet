"""SpecletEight model."""


from pathlib import Path
from typing import Optional

import pymc3 as pm
from pydantic import BaseModel
from theano import shared as ts

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import (
    CrisprScreenDataManager,
    Data,
    common_crispr_screen_transformations,
)
from src.modeling import feature_engineering as feng
from src.models.speclet_model import (
    ObservedVarName,
    SpecletModel,
    SpecletModelDataManager,
)


class SpecletEightConfiguration(BaseModel):
    """Configuration for SpecletEight."""

    broad_only: bool = True


def _append_total_read_counts(df: Data) -> Data:
    return achelp.append_total_read_counts(df)


def _add_useful_read_count_columns(df: Data) -> Data:
    return achelp.add_useful_read_count_columns(df)


class SpecletEight(SpecletModel):
    """## SpecletEight.

    A negative binomial model of the read counts from the CRISPR screen data.
    """

    _config: SpecletEightConfiguration

    def __init__(
        self,
        name: str,
        data_manager: Optional[SpecletModelDataManager] = None,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        config: Optional[SpecletEightConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletEight model.

        Args:
            name (str): A unique identifier for this instance of SpecletEight. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletEightConfiguration, optional): Model configuration.
        """
        logger.info("Creating a new SpecletEight object.")
        self._config = SpecletEightConfiguration() if config is None else config

        if data_manager is None:
            data_manager = CrisprScreenDataManager(
                data_source=DataFile.DEPMAP_CRC_SUBSAMPLE
            )

        data_transformations = common_crispr_screen_transformations.copy()
        data_transformations += [
            feng.zscale_rna_expression_by_gene_and_lineage,
            _append_total_read_counts,
            _add_useful_read_count_columns,
        ]
        if self._config.broad_only:
            data_transformations.append(achelp.filter_for_broad_source_only)

        # Need to add at end because `p_dna_batch` becomes non-categorical.
        data_transformations.append(achelp.set_achilles_categorical_columns)

        data_manager.add_transformation(data_transformations)

        super().__init__(name, data_manager, root_cache_dir=root_cache_dir, debug=debug)

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Define the PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
            ObservedVarName: Name of the target variable in the model.
        """
        logger.info("Creating SpecletEight model.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        logger.info(f"Number of data points: {total_size}")
        co_idx = achelp.common_indices(data)
        logger.info(f"Number of genes: {co_idx.n_genes}")
        logger.info(f"Number of cell lines: {co_idx.n_celllines}")
        logger.info(f"Number of lineages: {co_idx.n_lineages}")
        b_idx = achelp.data_batch_indices(data)

        logger.info("Creating shared variables.")
        x_initial = ts(data.counts_initial_adj.values)

        coords = {
            "sgrna": data.sgrna.cat.categories,
            "gene": data.hugo_symbol.cat.categories,
            "cell_line": data.depmap_id.cat.categories,
            "batch": data.p_dna_batch.cat.categories,
        }

        with pm.Model(coords=coords) as model:
            s = pm.Data("sgrna_idx", co_idx.sgrna_idx)
            # g = pm.Data("gene_idx", co_idx.gene_idx)
            c = pm.Data("cell_line_idx", co_idx.cellline_idx)
            b = pm.Data("batch_idx", b_idx.batch_idx)
            x_initial = pm.Data("x_initial", data.counts_initial_adj.values)
            counts_final = pm.Data("counts_final", data.counts_final.values)

            beta_sgrna = pm.Normal("beta_sgrna", 0, 2.5, dims=("sgrna", "cell_line"))

            mu_beta_batch = pm.Normal("mu_beta_batch", 0, 0.1)
            sigma_beta_batch = pm.HalfNormal("sigma_beta_batch", 0.1)
            beta_batch = pm.Normal(
                "beta_batch", mu_beta_batch, sigma_beta_batch, dims=("batch")
            )

            eta = pm.Deterministic("eta", beta_sgrna[s, c] + beta_batch[b])
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            alpha = pm.HalfNormal("alpha", 5)
            y = pm.NegativeBinomial(  # noqa: F841
                "y", x_initial * mu, alpha, observed=counts_final
            )

        return model, "y"
