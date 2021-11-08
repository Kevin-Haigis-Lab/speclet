"""SpecletNine model."""


from pathlib import Path
from typing import Any, Final, Optional

import pymc3 as pm
from pydantic import BaseModel
from pydantic.types import PositiveFloat  # noqa: F401

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
from src.project_enums import ModelParameterization, assert_never


class NormalParameters(BaseModel):
    """Normal distribution parameters."""

    mu: float
    sigma: PositiveFloat


class HalfNormalParameters(BaseModel):
    """Half-Normal distribution parameters."""

    sigma: PositiveFloat


class ExponentialParameters(BaseModel):
    """Exponential distribution parameters."""

    lam: PositiveFloat


class GammaParameters(BaseModel):
    """Gamma distribution parameters."""

    alpha: PositiveFloat
    beta: PositiveFloat


class SpecletNinePriors(BaseModel):
    """Priors for SpecletNine."""

    mu_beta: NormalParameters = NormalParameters(mu=0, sigma=2.5)
    sigma_beta: ExponentialParameters = ExponentialParameters(lam=2.5)
    alpha: GammaParameters = GammaParameters(alpha=2.0, beta=0.3)


class SpecletNineConfiguration(BaseModel):
    """Configuration for SpecletNine."""

    broad_only: bool = True
    beta_parameterization: ModelParameterization = ModelParameterization.CENTERED
    priors: SpecletNinePriors = SpecletNinePriors()


def make_speclet_nine_priors_config(
    mu_beta_mu: float,
    mu_beta_sigma: float,
    sigma_beta_lam: float,
    alpha_alpha: float,
    alpha_beta: float,
) -> SpecletNineConfiguration:
    """Generate a SpecletNine configuration with specific prior values.

    Args:
        mu_beta_mu (float): Centrality of `mu_beta`.
        mu_beta_sigma (float): Standard deviation of `mu_beta`.
        sigma_beta_lam (float): Lambda of `sigma_beta`.
        alpha_alpha (float): Alpha of `alpha`.
        alpha_beta (float): Beta of `alpha`.

    Returns:
        SpecletNineConfiguration: SpecletNineConfiguration with the specific priors.
    """
    priors = SpecletNinePriors(
        mu_beta=NormalParameters(mu=mu_beta_mu, sigma=mu_beta_sigma),
        sigma_beta=ExponentialParameters(lam=sigma_beta_lam),
        alpha=GammaParameters(alpha=alpha_alpha, beta=alpha_beta),
    )
    return SpecletNineConfiguration(priors=priors)


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

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        logger.info("Setting model configuration.")
        self._config = SpecletNineConfiguration(**info)
        return None

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
            "lineage": data.lineage.cat.categories,
        }

        priors = self._config.priors
        logger.info(f"Model prior constants: {priors.dict()}")
        beta_param = self._config.beta_parameterization
        logger.info(f"Beta parameterization: {beta_param.value}")

        logger.info("Building PyMC3 model.")
        with pm.Model(coords=coords) as model:
            g = pm.Data("gene_idx", co_idx.gene_idx)
            c = pm.Data("cell_line_idx", co_idx.cellline_idx)
            # l_c = pm.Data("cell_line_to_lineage_idx", co_idx.cellline_to_lineage_idx)
            ct_initial = pm.Data("ct_initial", data.counts_initial_adj.values)
            ct_final = pm.Data("ct_final", data.counts_final.values)

            mu_beta = pm.Normal("mu_beta", priors.mu_beta.mu, priors.mu_beta.sigma)
            sigma_beta = pm.Exponential("sigma_beta", priors.sigma_beta.lam)

            if beta_param is ModelParameterization.CENTERED:
                beta = pm.Normal(
                    "beta", mu_beta, sigma_beta, dims=("gene", "cell_line")
                )
            elif beta_param is ModelParameterization.NONCENTERED:
                delta_beta = pm.Normal("delta_beta", 0, 1, dims=("gene", "cell_line"))
                beta = pm.Deterministic("beta", mu_beta + delta_beta * sigma_beta)
            else:
                assert_never(beta_param)

            eta = pm.Deterministic("eta", beta[g, c])
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            alpha = pm.Gamma("alpha", priors.alpha.alpha, priors.alpha.beta)
            y = pm.NegativeBinomial(  # noqa: F841
                "y", ct_initial * mu, alpha, observed=ct_final
            )

        return model, "y"
