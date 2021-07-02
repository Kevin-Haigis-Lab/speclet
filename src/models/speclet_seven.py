"""Speclet Model Six."""

from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import pymc3 as pm
from theano import shared as ts
from theano import tensor
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel
from src.project_enums import ModelParameterization as MP


class SpecletSevenParameterization(NamedTuple):
    """Parameterizations for each covariate in SpecletSeven model."""

    a: MP = MP.CENTERED
    μ_a: MP = MP.CENTERED
    μ_μ_a: MP = MP.CENTERED


class SpecletSeven(SpecletModel):
    """SpecletSeven Model.

    $$
    \\begin{aligned}
    lfc &\\sim N(\\mu, \\sigma) \\\\
    \\mu &= a_{s,c} \\quad \\sigma \\sim HN(1) \\\\
    a_{s,c} &\\sim N(\\mu_a, \\sigma_a)_{s,c} \\\\
    \\mu_a &\\sim N(\\mu_{\\mu_a}, \\sigma_{\\mu_a})_{g,c}
      \\quad \\sigma_a \\sim HN(\\sigma_{\\sigma_a})_{s}
      \\quad \\sigma_{\\sigma_a} \\sim HN(1) \\\\
    \\mu_{\\mu_a} &\\sim N(\\mu_{\\mu_{\\mu_a}}, \\sigma_{\\mu_{\\mu_a}})_{g,l}
      \\quad \\sigma_{\\mu_a} \\sim HN(\\sigma_{\\sigma_{\\mu_a}})_{c}
      \\quad \\sigma_{\\sigma_{\\mu_a}} \\sim HN(1)_{c} \\\\
    \\mu_{\\mu_{\\mu_a}} &\\sim N(0, 1)
      \\quad \\sigma_{\\mu_{\\mu_a}} \\sim HN(1)
    \\end{aligned}
    $$

    where:

    - s: sgRNA
    - g: gene
    - c: cell line
    - l: cell line lineage

    A very deep hierarchical model that is, in part, meant to be a proof-of-concept for
    constructing, fitting, and interpreting such a "tall" hierarchical model.

    Attributes:
        noncentered_param (bool): Use the non-centered parameterization.
    """

    _noncentered_param: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        parameterization: SpecletSevenParameterization = SpecletSevenParameterization(),
    ) -> None:
        """Instantiate a SpecletSeven model.

        Args:
            name (str): A unique identifier for this instance of SpecletSeven. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            noncentered_param (bool, optional): Use the non-centered parameterization.
              Defaults to False.
            parameterization (SpecletFiveParameterization, optional): Covariate-specific
              parameterization options.
        """
        logger.debug("Instantiating a SpecletSeven model.")
        if data_manager is None:
            logger.debug("Creating a data manager since none was supplied.")
            data_manager = CrcDataManager(debug=debug)

        self.parameterization = parameterization

        super().__init__(
            name="speclet-six_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def _model_specification(
        self,
        co_idx: achelp.CommonIndices,
        sgrna_idx_shared: TTShared,
        sgrna_to_gene_idx_shared: TTShared,
        cellline_idx_shared: TTShared,
        cellline_to_lineage_idx_shared: TTShared,
        lfc_shared: TTShared,
        total_size: int,
    ) -> Tuple[pm.Model, str]:
        _mu_a_shape = (co_idx.n_genes, co_idx.n_celllines)
        with pm.Model() as model:
            μ_μ_μ_a = pm.Normal("μ_μ_μ_a", 0, 2)
            σ_μ_μ_a = pm.HalfNormal("σ_μ_μ_a", 2)

            σ_σ_μ_a = pm.HalfNormal("σ_σ_μ_a", 1)
            σ_μ_a = pm.HalfNormal("σ_μ_a", σ_σ_μ_a, shape=(1, co_idx.n_celllines))

            σ_σ_a = pm.HalfNormal("σ_σ_a", 1)
            σ_a = pm.HalfNormal("σ_a", σ_σ_a, shape=(co_idx.n_sgrnas, 1))

            if self.parameterization.μ_μ_a is MP.NONCENTERED:
                μ_μ_a_offset = pm.Normal(
                    "μ_μ_a_offset", 0, 1.0, shape=(co_idx.n_genes, co_idx.n_lineages)
                )
                μ_μ_a = pm.Deterministic("μ_μ_a", μ_μ_μ_a + μ_μ_a_offset * σ_μ_μ_a)
            else:
                μ_μ_a = pm.Normal(
                    "μ_μ_a",
                    μ_μ_μ_a,
                    σ_μ_μ_a,
                    shape=(co_idx.n_genes, co_idx.n_lineages),
                )

            if self.parameterization.μ_a is MP.NONCENTERED:
                μ_a_offset = pm.Normal("μ_a_offset", 0, 1.0, shape=_mu_a_shape)
                μ_a = pm.Deterministic(
                    "μ_a",
                    μ_μ_a[:, cellline_to_lineage_idx_shared] + μ_a_offset * σ_μ_a,
                )
            else:
                μ_a = pm.Normal(
                    "μ_a",
                    μ_μ_a[:, cellline_to_lineage_idx_shared],
                    tensor.ones(shape=_mu_a_shape) * σ_μ_a,
                    shape=_mu_a_shape,
                )

            if self.parameterization.a is MP.NONCENTERED:
                a_offset = pm.Normal(
                    "a_offset", 0, 1.0, shape=(co_idx.n_sgrnas, co_idx.n_celllines)
                )
                a = pm.Deterministic(
                    "a", μ_a[sgrna_to_gene_idx_shared, :] + a_offset * σ_a
                )
            else:
                a = pm.Normal(
                    "a",
                    μ_a[sgrna_to_gene_idx_shared, :],
                    σ_a,
                    shape=(co_idx.n_sgrnas, co_idx.n_celllines),
                )

            μ = pm.Deterministic("μ", a[sgrna_idx_shared, cellline_idx_shared])

            # Standard deviation of log-fold change.
            σ = pm.HalfNormal("σ", 1)

            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ,
                observed=lfc_shared,
                total_size=total_size,
            )

        return model, "lfc"

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletSeven model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = ts(co_idx.sgrna_idx)
        sgrna_to_gene_idx_shared = ts(co_idx.sgrna_to_gene_idx)
        cellline_idx_shared = ts(co_idx.cellline_idx)
        cellline_to_lineage_idx_shared = ts(co_idx.cellline_to_lineage_idx)
        lfc_shared = ts(data.lfc.values)

        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "cellline_to_lineage_idx_shared": cellline_to_lineage_idx_shared,
            "lfc_shared": lfc_shared,
        }

        logger.info("Creating PyMC3 model for SpecletSeven.")
        model, obs_var_name = self._model_specification(
            co_idx=co_idx,
            sgrna_idx_shared=sgrna_idx_shared,
            sgrna_to_gene_idx_shared=sgrna_to_gene_idx_shared,
            cellline_idx_shared=cellline_idx_shared,
            cellline_to_lineage_idx_shared=cellline_to_lineage_idx_shared,
            lfc_shared=lfc_shared,
            total_size=total_size,
        )
        logger.debug("Finished building model.")
        return model, obs_var_name

    def get_replacement_parameters(self) -> ReplacementsDict:
        """Make a dictionary mapping the shared data variables to new data.

        Raises:
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        logger.debug("Making dictionary of replacement parameters.")

        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        data = self.data_manager.get_data()
        mb_size = self.data_manager.get_batch_size()
        co_idx = achelp.common_indices(data)

        sgrna_idx_batch = pm.Minibatch(co_idx.sgrna_idx, batch_size=mb_size)
        cellline_idx_batch = pm.Minibatch(co_idx.cellline_idx, batch_size=mb_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=mb_size)

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

    def update_mcmc_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        logger.info("Updating the MCMC sampling parameters.")
        self.mcmc_sampling_params.draws = 4000
        self.mcmc_sampling_params.tune = 2000
        self.mcmc_sampling_params.target_accept = 0.99
        return None

    def update_advi_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        logger.info("Updating the ADVI fitting parameters.")
        parameter_adjustment_map = {True: 40000, False: 100000}
        self.advi_sampling_params.n_iterations = parameter_adjustment_map[self.debug]
        return None
