"""Helpers for organizing simualtaion-based calibrations."""

import math
import re
from enum import Enum, unique
from pathlib import Path
from random import choices
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.exceptions
from src.data_processing import achilles as achelp
from src.data_processing import vectors as vhelp
from src.io.data_io import DataFile, data_path
from src.modeling.pymc3_analysis import get_hdi_colnames_from_az_summary
from src.string_functions import prefixed_count

#### ---- File management ---- ####


class SBCResults:
    """Results from a single round of SBC."""

    priors: Dict[str, Any]
    inference_obj: az.InferenceData
    posterior_summary: pd.DataFrame

    def __init__(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ):
        """Create an instance of SBCResults.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        self.priors = priors
        self.inference_obj = inference_obj
        self.posterior_summary = posterior_summary


class SBCFileManager:
    """Manages the results from a round of simulation-based calibration."""

    dir: Path
    sbc_data_path: Path
    inference_data_path: Path
    priors_path_set: Path
    priors_path_get: Path
    posterior_summary_path: Path

    sbc_data: Optional[pd.DataFrame] = None
    sbc_results: Optional[SBCResults] = None

    def __init__(self, dir: Path):
        """Create a SBCFileManager.

        Args:
            dir (Path): The directory where the data is stored.
        """
        if not dir.is_dir():
            raise NotADirectoryError(dir)
        self.dir = dir
        self.sbc_data_path = dir / "sbc-data.csv"
        self.inference_data_path = dir / "inference-data.netcdf"
        self.priors_path_set = dir / "priors"
        self.priors_path_get = dir / "priors.npz"
        self.posterior_summary_path = dir / "posterior-summary.csv"
        _ = self._check_dir_exists()

    def _check_dir_exists(self) -> bool:
        if self.dir.exists():
            return True
        else:
            self.dir.mkdir()
            return False

    def get_sbc_data(self, re_read: bool = False) -> pd.DataFrame:
        """Get the simulated data for the SBC.

        Args:
            re_read (bool, optional): Force re-reading from file. Defaults to False.

        Returns:
            pd.DataFrame: Saved simulated data frame.
        """
        if self.sbc_data is None or re_read:
            self.sbc_data = pd.read_csv(self.sbc_data_path)
        return self.sbc_data

    def save_sbc_data(
        self, data: pd.DataFrame, index: bool = False, **kwargs: dict[str, Any]
    ) -> None:
        """Save SBC dataframe to disk.

        Args:
            data (pd.DataFrame): Simulated data used for the SBC.
            index (bool, optional): Should the index be included in the file (CSV).
              Defaults to False.
            kwargs (dict[str, Any]): Additional keyword arguments for `data.to_csv()`.
        """
        self.sbc_data = data
        data.to_csv(self.sbc_data_path, index=index)

    def simulation_data_exists(self) -> bool:
        """Does the simulation dataframe file exist?

        Returns:
            bool: True if the dataframe file exists.
        """
        return self.sbc_data_path.exists()

    def clear_saved_data(self) -> None:
        """Clear save SBC dataframe file."""
        if self.simulation_data_exists():
            self.sbc_data_path.unlink()

    def save_sbc_results(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ) -> None:
        """Save the results from a round of SBC.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        inference_obj.to_netcdf(self.inference_data_path.as_posix())
        np.savez(self.priors_path_set.as_posix(), **priors)
        posterior_summary.to_csv(
            self.posterior_summary_path.as_posix(), index_label="parameter"
        )

    def _tidy_numpy_files(self, files: Any) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        for k in files.files:
            d[k] = files[k]
        return d

    def get_sbc_results(self, re_read: bool = False) -> SBCResults:
        """Retrieve results of a round of SBC.

        Args:
            re_read (bool, optional): Should the results be re-read from file?
              Defaults to False.

        Returns:
            SBCResults: The results from the round of SBC.
        """
        if self.sbc_results is not None and not re_read:
            return self.sbc_results

        inference_obj = az.from_netcdf(self.inference_data_path)
        priors_files = np.load(self.priors_path_get.as_posix())
        priors = self._tidy_numpy_files(priors_files)
        posterior_summary = pd.read_csv(
            self.posterior_summary_path, index_col="parameter"
        )

        self.sbc_results = SBCResults(
            priors=priors,
            inference_obj=inference_obj,
            posterior_summary=posterior_summary,
        )
        return self.sbc_results

    def all_data_exists(self) -> bool:
        """Confirm that all data exists.

        Returns:
            bool: True if all of the data exists, else false.
        """
        for p in [
            self.priors_path_get,
            self.posterior_summary_path,
            self.inference_data_path,
        ]:
            if not p.exists():
                return False
        return True

    def clear_results(self) -> None:
        """Clear the stored SBC results (if they exist)."""
        for f in (
            self.inference_data_path,
            self.priors_path_get,
            self.posterior_summary_path,
        ):
            if f.exists():
                f.unlink()


#### ---- Results analysis ---- ####


class SBCAnalysis:
    """Analysis of SBC results."""

    root_dir: Path
    pattern: str
    n_simulations: Optional[int]

    def __init__(
        self, root_dir: Path, pattern: str, n_simulations: Optional[int] = None
    ) -> None:
        """Create a `SBCAnalysis` object.

        Args:
            root_dir (Path): Path to the directory containing the results of all of
              the simulations.
            pattern (str): Pattern used for naming the simulations.
            n_simulations (Optional[int], optional): Number of simulations expected. If
              supplied, this number will be used to check the root dir for all of the
              results. Defaults to None.
        """
        self.root_dir = root_dir
        self.pattern = pattern
        self.n_simulations = n_simulations

    def _check_n_sims(self, ls: Sequence[Any]) -> None:
        if self.n_simulations is not None and self.n_simulations != len(ls):
            raise src.exceptions.IncorrectNumberOfFilesFoundError(
                expected=self.n_simulations, found=len(ls)
            )

    def get_simulation_directories(self) -> list[Path]:
        """Get the directories of all of the simulation results.

        Returns:
            list[Path]: List of paths.
        """
        return [p for p in self.root_dir.iterdir() if self.pattern in p.name]

    def get_simulation_file_managers(self) -> list[SBCFileManager]:
        """Get the file managers for each simulation.

        Returns:
            list[SBCFileManager]: List of SBC file managers.
        """
        return [SBCFileManager(p) for p in self.get_simulation_directories()]

    def get_simulation_results(self) -> list[SBCResults]:
        """Get all of the simulation results.

        Raises:
            src.exceptions.CacheDoesNotExistError: Raised if the results do not exist
            for a simulation.

        Returns:
            list[SBCResults]: List of all simulation results.
        """
        fms = self.get_simulation_file_managers()
        results: list[SBCResults] = []
        for fm in fms:
            if fm.all_data_exists():
                results.append(fm.get_sbc_results())
            else:
                raise src.exceptions.CacheDoesNotExistError(fm.dir)

        self._check_n_sims(results)
        return results

    def posterior_accuracy(
        self,
        simulation_posteriors_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Get the accuracy of the simulation results.

        Args:
            simulation_posteriors_df (Optional[pd.DataFrame], optional): Data frame of
            the summaries of the posterior summaries. If None is provided, then the data
            frame will be made first. Defaults to None.

        Returns:
            pd.DataFrame: Data frame of the accuracy of each parameter in the model.
        """
        if simulation_posteriors_df is None:
            simulation_posteriors_df = collate_sbc_posteriors(
                posterior_dirs=self.get_simulation_directories(),
                num_permutations=self.n_simulations,
            )
        return (
            simulation_posteriors_df.copy()
            .groupby(["parameter_name"])["within_hdi"]
            .mean()
            .reset_index(drop=False)
            .sort_values("within_hdi", ascending=False)
            .reset_index(drop=True)
        )

    def uniformity_test(self) -> pd.DataFrame:
        """Perform the uniformity SBC analysis.

        Returns:
            pd.DataFrame: Rank statistics for each parameter.
        """
        sbc_file_managers = self.get_simulation_file_managers()
        self._check_n_sims(sbc_file_managers)
        unformity_results: list[pd.DataFrame] = []
        for sbc_fm in sbc_file_managers:
            unformity_results.append(
                calculate_parameter_rank_statistic(sbc_fm.get_sbc_results())
            )

        return pd.DataFrame()


def _thin_posterior(ary: np.ndarray, every_other: int) -> np.ndarray:
    _s = ary.shape
    if len(_s) == 1:
        return ary[::every_other]
    elif len(_s) == 2:
        return ary[::every_other, ::every_other]
    else:
        raise NotImplementedError(
            f"Thinning not implemented for a variable with {_s} dimensions."
        )


def _array_to_indexed_dataframe(ary: np.ndarray, name: str) -> pd.DataFrame:
    p: list[str] = []
    v: list[int] = []
    if ary.shape == (1,):
        p.append(name)
        v.append(ary[0])
    elif len(ary.shape) == 1:
        for i in range(ary.shape[0]):
            p.append(f"{name}[{i}]")
            v.append(ary[i])
    elif len(ary.shape) == 2:
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                p.append(f"{name}[{i},{j}]")
                v.append(ary[i, j])
    else:
        raise NotImplementedError(
            "Turning an array into a data frame is not implemented for >2D arrays."
        )

    return pd.DataFrame({"parameter": p, "rank_stat": v})


def calculate_parameter_rank_statistic(
    sbc_res: SBCResults, thinning: int = 1
) -> pd.DataFrame:
    """Calculate the rank statistics for SBC uniformity analysis.

    Args:
        sbc_res (SBCResults): The results of an SBC simulation.
        thinning (int, optional): How to thin the posterior estimates. A value of 1
          results in no thinning, a value of 2 results in thinning to every other value,
          etc. Defaults to 1.

    Returns:
        pd.DataFrame: A data frame with the rank statistics for each parameter.
    """
    rank_df = pd.DataFrame()
    for name, theta in sbc_res.priors.items():
        theta_tilde = sbc_res.inference_obj["posterior"][name].values
        theta_tilde = _thin_posterior(theta_tilde, every_other=thinning)
        rank_scores = (theta < theta_tilde).sum(axis=0)
        df = _array_to_indexed_dataframe(rank_scores, name=name)
        rank_df = pd.concat([rank_df, df])
    return rank_df


#### ---- Mock data generation ---- ####


def generate_mock_sgrna_gene_map(n_genes: int, n_sgrnas_per_gene: int) -> pd.DataFrame:
    """Generate a fake sgRNA-gene map.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNA per gene.

    Returns:
        pd.DataFrame: A data frame mapping each sgRNA to a gene. Each sgRNA only matches
          to a single gene and each gene will have `n_sgrnas_per_gene` sgRNAs mapped
          to it.
    """
    genes = prefixed_count("gene", n=n_genes)
    sgrna_target_chr = choices(["Chr1", "Chr2", "Chr3"], k=n_genes)
    sgrnas = [prefixed_count(gene + "_sgrna", n=n_sgrnas_per_gene) for gene in genes]
    return pd.DataFrame(
        {
            "hugo_symbol": np.repeat(genes, n_sgrnas_per_gene),
            "sgrna_target_chr": np.repeat(sgrna_target_chr, n_sgrnas_per_gene),
            "sgrna": np.array(sgrnas).flatten(),
        }
    )


@unique
class SelectionMethod(str, Enum):
    """Methods for selecting `n` elements from a list."""

    random = "random"
    tiled = "tiled"
    repeated = "repeated"
    shuffled = "shuffled"


def select_n_elements_from_l(
    n: int, list: Union[List[Any], np.ndarray], method: Union[SelectionMethod, str]
) -> np.ndarray:
    """Select `n` elements from a collection `l` using a specified method.

    There are three available methods:

    1. `random`: Randomly select `n` values from `l`.
    2. `tiled`: Use `numpy.tile()` (`[1, 2, 3]` → `[1, 2, 3, 1, 2, 3, ...]`).
    3. `repeated`: Use `numpy.repeat()` (`[1, 2, 3]` → `[1, 1, 2, 2, 3, 3, ...]`).
    4. `shuffled`: Shuffles the results of `numpy.tile()` to get even, random coverage.

    Args:
        n (int): Number elements to draw.
        l (Union[List[Any], np.ndarray]): Collection to draw from.
        method (SelectionMethod): Method to use for drawing elements.

    Raises:
        ValueError: Raised if an unknown method is passed.

    Returns:
        np.ndarray: A numpy array of length 'n' with values from 'l'.
    """
    if isinstance(method, str):
        method = SelectionMethod(method)

    size = math.ceil(n / len(list))

    if method == SelectionMethod.random:
        return np.random.choice(list, n)
    elif method == SelectionMethod.tiled:
        return np.tile(list, size)[:n]
    elif method == SelectionMethod.repeated:
        return np.repeat(list, size)[:n]
    elif method == SelectionMethod.shuffled:
        a = np.tile(list, size)[:n]
        np.random.shuffle(a)
        return a
    else:
        raise ValueError(f"Unknown selection method: {method}")


def generate_mock_cell_line_information(
    genes: Union[List[str], np.ndarray],
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool = False,
) -> pd.DataFrame:
    """Generate mock "sample information" for fake cell lines.

    Args:
        genes (List[str]): List of genes tested in the cell lines.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.
        randomness (bool, optional): Should the lineages, screens, and batches be
          randomly assigned or applied in a pattern? Defaults to False (patterned).

    Returns:
        pd.DataFrame: The mock sample information.
    """
    # Methods for selecting elements from the list to produce pairings.
    _lineage_method = "random" if randomness else "tiled"
    _batch_method = "random" if randomness else "shuffled"
    _screen_method = "random" if randomness else "tiled"

    cell_lines = prefixed_count("cellline", n=n_cell_lines)
    lineages = prefixed_count("lineage", n=n_lineages)
    batches = prefixed_count("batch", n=n_batches)
    batch_map = pd.DataFrame(
        {
            "depmap_id": cell_lines,
            "lineage": select_n_elements_from_l(
                n_cell_lines, lineages, _lineage_method
            ),
            "p_dna_batch": select_n_elements_from_l(
                n_cell_lines, batches, _batch_method
            ),
        }
    )

    screens = ["broad"]
    if n_screens == 2:
        screens += ["sanger"]
    if n_screens > 2:
        screens += prefixed_count("screen", n=n_screens - 2)

    screen_map = pd.DataFrame(
        {
            "p_dna_batch": batches,
            "screen": select_n_elements_from_l(n_batches, screens, _screen_method),
        }
    )

    return (
        pd.DataFrame(
            {
                "depmap_id": np.repeat(cell_lines, len(np.unique(genes))),
                "hugo_symbol": np.tile(genes, n_cell_lines),
            }
        )
        .merge(batch_map, on="depmap_id")
        .merge(screen_map, on="p_dna_batch")
    )


def generate_mock_achilles_categorical_groups(
    n_genes: int,
    n_sgrnas_per_gene: int,
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
    randomness: bool = False,
) -> pd.DataFrame:
    """Generate mock Achilles categorical column scaffolding.

    This function should be used to generate a scaffolding of the Achilles data. It
    creates columns that mimic the hierarchical natrue of the Achilles categorical
    columns. Each sgRNA maps to a single gene. Each cell lines only received on pDNA
    batch. Each cell line / sgRNA combination occurs exactly once.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNAs per gene.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.
        randomness (bool, optional): Should the lineages, screens, and batches be
          randomly assigned or applied in a pattern? Defaults to False (patterned).

    Returns:
        pd.DataFrame: A pandas data frame the resembles the categorical column
        hierarchical structure of the Achilles data.
    """
    sgnra_map = generate_mock_sgrna_gene_map(
        n_genes=n_genes, n_sgrnas_per_gene=n_sgrnas_per_gene
    )
    cell_line_info = generate_mock_cell_line_information(
        genes=sgnra_map.hugo_symbol.unique(),
        n_cell_lines=n_cell_lines,
        n_lineages=n_lineages,
        n_batches=n_batches,
        n_screens=n_screens,
        randomness=randomness,
    )

    def _make_cat_cols(_df: pd.DataFrame) -> pd.DataFrame:
        return achelp.set_achilles_categorical_columns(_df, cols=_df.columns.tolist())

    return (
        cell_line_info.merge(sgnra_map, on="hugo_symbol")
        .reset_index(drop=True)
        .pipe(_make_cat_cols)
    )


def _make_mock_grouped_copy(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]]
) -> pd.DataFrame:
    df_copy = mock_df.copy()
    if grouping_cols is not None:
        df_copy = df_copy[grouping_cols].drop_duplicates()
    return df_copy


def _merge_mock_and_grouped_copy(
    mock_df: pd.DataFrame, df_copy: pd.DataFrame, grouping_cols: Optional[list[str]]
) -> pd.DataFrame:
    if grouping_cols is not None:
        return mock_df.merge(df_copy, left_index=False, right_index=False)
    return df_copy


def add_mock_copynumber_data(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Add mock copy number data to mock Achilles data.

    The mock CNA values actually come from real copy number values from CRC cancer cell
    lines. The values are randomly sampled with replacement and some noise is added to
    each value.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same CN value. Defaults to
          None for no group effect.

    Returns:
        pd.DataFrame: Same mock Achilles data frame with a new "copy_number" column.
    """
    real_cna_values = np.load(data_path(DataFile.copy_number_sample))

    df_copy = _make_mock_grouped_copy(mock_df, grouping_cols)
    mock_cn = np.random.choice(real_cna_values, size=df_copy.shape[0], replace=True)
    mock_cn = mock_cn + np.random.normal(0, 0.1, size=mock_cn.shape)
    mock_cn = vhelp.squish_array(mock_cn, lower=0.0, upper=20.0)
    df_copy["copy_number"] = mock_cn.flatten()
    return _merge_mock_and_grouped_copy(mock_df, df_copy, grouping_cols)


def add_mock_rna_expression_data(
    mock_df: pd.DataFrame,
    grouping_cols: Optional[list[str]] = None,
    subgroups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add fake RNA expression data to a mock Achilles data frame.

    The RNA expression values are sampled from a normal distribution with mean and
    standard deviation that are each sampled from different normal distributions. If a
    grouping is supplied, then each value in the group will be sampled from the same
    distribution (i.e. same mean and standard deviation).

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same RNA value. Defaults to
          None for no group effect.
        subgroups (Optional[list[str]], optional): List of columns to group by. Each
          group will have the same mean and standard deviation for the sampling
          distribution. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """

    def _rna_normal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = _make_mock_grouped_copy(df, grouping_cols)
        mu = np.abs(np.random.normal(10.0, 3))
        sd = np.abs(np.random.normal(0.0, 3))
        rna_expr = np.random.normal(mu, sd, size=df_copy.shape[0])
        rna_expr = vhelp.squish_array(rna_expr, lower=0.0, upper=np.inf)
        df_copy["rna_expr"] = rna_expr
        df_copy = _merge_mock_and_grouped_copy(df, df_copy, grouping_cols)
        return df_copy

    if subgroups is None:
        mock_df = _rna_normal_distribution(mock_df)
    else:
        mock_df = (
            mock_df.groupby(subgroups)
            .apply(_rna_normal_distribution)
            .reset_index(drop=True)
        )
    return mock_df


def add_mock_is_mutated_data(
    mock_df: pd.DataFrame, grouping_cols: Optional[list[str]] = None, prob: float = 0.01
) -> pd.DataFrame:
    """Add a mutation column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        grouping_cols (Optional[list[str]], optional): Columns to group by where every
          appearance of the same combination will have the same mutation value. Defaults
          to None for no group effect.
        prob (float, optional): The probability of a gene being mutated. All mutations
          are indpendent of each other. Defaults to 0.01.

    Returns:
        pd.DataFrame: The same mock Achilles data frame with an "is_mutated" columns.
    """
    df_copy = _make_mock_grouped_copy(mock_df, grouping_cols)
    df_copy["is_mutated"] = np.random.uniform(0, 1, size=df_copy.shape[0]) < prob
    return _merge_mock_and_grouped_copy(mock_df, df_copy, grouping_cols)


def add_mock_zero_effect_lfc_data(
    mock_df: pd.DataFrame, mu: float = 0.0, sigma: float = 0.5
) -> pd.DataFrame:
    """Add fake log-fold change column to mock Achilles data.

    Args:
        mock_df (pd.DataFrame): Mock Achilles data frame.
        mu (float, optional): Mean of normal distribution for sampling LFC values.
          Defaults to 0.0.
        sigma (float, optional): Standard deviation of normal distribution for sampling
          LFC values. Defaults to 0.5.

    Returns:
        pd.DataFrame: Same mock Achilles data with a new "lfc" column.
    """
    mock_df["lfc"] = np.random.normal(mu, sigma, mock_df.shape[0])
    return mock_df


def generate_mock_achilles_data(
    n_genes: int,
    n_sgrnas_per_gene: int,
    n_cell_lines: int,
    n_lineages: int,
    n_batches: int,
    n_screens: int,
) -> pd.DataFrame:
    """Generate mock Achilles data.

    Each sgRNA maps to a single gene. Each cell lines only received on pDNA batch.
    Each cell line / sgRNA combination occurs exactly once.

    Args:
        n_genes (int): Number of genes.
        n_sgrnas_per_gene (int): Number of sgRNAs per gene.
        n_cell_lines (int): Number of cell lines.
        n_lineages (int, optional): Number of lineages. Must be less than or equal to
          the number of cell lines.
        n_batches (int): Number of pDNA batchs.
        n_screens (int): Number of screens sourced for the data. Must be less than or
          equal to the number of batches.

    Returns:
        pd.DataFrame: A pandas data frame the resembles the Achilles data.
    """
    return (
        generate_mock_achilles_categorical_groups(
            n_genes=n_genes,
            n_sgrnas_per_gene=n_sgrnas_per_gene,
            n_cell_lines=n_cell_lines,
            n_lineages=n_lineages,
            n_batches=n_batches,
            n_screens=n_screens,
        )
        .pipe(add_mock_copynumber_data, grouping_cols=["hugo_symbol", "depmap_id"])
        .pipe(
            add_mock_rna_expression_data,
            grouping_cols=["hugo_symbol", "depmap_id"],
            subgroups=["hugo_symbol", "lineage"],
        )
        .pipe(add_mock_is_mutated_data, grouping_cols=["hugo_symbol", "depmap_id"])
        .pipe(add_mock_zero_effect_lfc_data)
    )


#### ---- Collate SBC ---- ####


def _split_parameter(p: str) -> List[str]:
    return [a for a in re.split("\\[|,|\\]", p) if a != ""]


def _get_prior_value_using_index_list(ary: np.ndarray, idx: List[int]) -> float:
    """Extract a prior value from an array by indexes in a list.

    The input array may be any number of dimensions and the indices in the list each
    correspond to a single dimension. The final result will be a single value extract
    from the array.

    Args:
        ary (np.ndarray): Input array of priors of any dimension.
        idx (List[int]): List of indices, one per dimension.

    Returns:
        float: The value in the array at a location.
    """
    if ary.shape == (1,):
        ary = np.asarray(ary[0])
    res = vhelp.index_array_by_list(ary, idx)
    assert len(res.shape) == 0
    return float(res)


def _make_priors_dataframe(
    priors: Dict[str, np.ndarray], parameters: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame({"parameter": parameters, "true_value": 0}).set_index("parameter")
    for parameter in parameters:
        split_p = _split_parameter(parameter)
        param = split_p[0]
        idx = [int(i) for i in split_p[1:]]
        value = _get_prior_value_using_index_list(priors[param][0], idx)
        df.loc[parameter] = value
    return df


def _is_true_value_within_hdi(
    low_hdi: pd.Series, true_vals: pd.Series, high_hdi: pd.Series
) -> np.ndarray:
    return (
        (low_hdi.values < true_vals.values).astype(int)
        * (true_vals.values < high_hdi.values).astype(int)
    ).astype(bool)


def _assign_column_for_within_hdi(
    df: pd.DataFrame, true_value_col: str = "true_value"
) -> pd.DataFrame:
    hdi_low, hdi_high = get_hdi_colnames_from_az_summary(df)
    df["within_hdi"] = _is_true_value_within_hdi(
        df[hdi_low], df["true_value"], df[hdi_high]
    )
    return df


class SBCResultsNotFoundError(FileNotFoundError):
    """SBC Results not found."""

    pass


def get_posterior_summary_for_file_manager(sbc_dir: Path) -> pd.DataFrame:
    """Create a summary of the results of an SBC sim. cached in a directory.

    Args:
        sbc_dir (Path): Directory with the results of a SBC simulation.

    Raises:
        SBCResultsNotFoundError: Raised if the SBC results are not found.

    Returns:
        pd.DataFrame: Dataframe with the summary of the simulation results.
    """
    sbc_fm = SBCFileManager(sbc_dir)
    if not sbc_fm.all_data_exists():
        raise SBCResultsNotFoundError(f"Not all output from '{sbc_fm.dir.name}' exist.")
    res = sbc_fm.get_sbc_results()
    true_values = _make_priors_dataframe(
        res.priors, parameters=res.posterior_summary.index.values.tolist()
    )
    return res.posterior_summary.merge(true_values, left_index=True, right_index=True)


def _index_and_concat_summaries(post_summaries: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            d.assign(simulation_id=f"sim_id_{str(i).rjust(4, '0')}")
            for i, d in enumerate(post_summaries)
        ]
    )


def _extract_parameter_names(df: pd.DataFrame) -> pd.DataFrame:
    df["parameter_name"] = [x.split("[")[0] for x in df.index.values]
    df = df.set_index("parameter_name", append=True)
    return df


def collate_sbc_posteriors(
    posterior_dirs: Iterable[Path], num_permutations: Optional[int] = None
) -> pd.DataFrame:
    """Collate many SBC posteriors.

    Args:
        posterior_dirs (Iterable[Path]): The directories containing the stored results
          of many SBC simulations.
        num_permutations (Optional[int], optional): Number of permutations expected. If
          supplied, this will be checked against the number of found simulations.
          Defaults to None.

    Raises:
        IncorrectNumberOfFilesFoundError: Raised if the number of found simulations is
        not equal to the number of expected simulations.

    Returns:
        pd.DataFrame: A single data frame with all of the results of the simulations.
    """
    tqdm_posterior_dirs = tqdm(posterior_dirs, total=num_permutations)
    simulation_posteriors: List[pd.DataFrame] = [
        get_posterior_summary_for_file_manager(d) for d in tqdm_posterior_dirs
    ]

    if num_permutations is not None and len(simulation_posteriors) != num_permutations:
        raise src.exceptions.IncorrectNumberOfFilesFoundError(
            num_permutations, len(simulation_posteriors)
        )

    simulation_posteriors_df = (
        _index_and_concat_summaries(simulation_posteriors)
        .pipe(_extract_parameter_names)
        .pipe(_assign_column_for_within_hdi)
    )

    return simulation_posteriors_df
