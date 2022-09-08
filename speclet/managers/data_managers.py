"""Data management classes."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from speclet.data_processing.crispr import (
    set_achilles_categorical_columns,
    set_chromosome_categories,
)
from speclet.data_processing.validation import (
    check_between,
    check_finite,
    check_nonnegative,
    check_unique_groups,
)
from speclet.exceptions import (
    DataFileDoesNotExist,
    DataFileIsNotAFile,
    UnsupportedDataFileType,
)
from speclet.io import DataFile, data_path, modeling_data_dir
from speclet.utils.general import merge_sets

data_transformation = Callable[[pd.DataFrame], pd.DataFrame]

SUPPORTED_DATA_FILES: Final[set[str]] = {".csv", ".tsv", ".pkl"}

# Some common CRISPR data transformation functions.


def broad_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for only Broad data.

    Note that the index is reset before returning.
    """
    return df[df["screen"] == "broad"].reset_index(drop=True)


class CrisprScreenDataManager:
    """Manage CRISPR screen data."""

    def __init__(
        self,
        data_file: Path | DataFile | str,
        transformations: list[data_transformation] | None = None,
    ) -> None:
        """Create a CRISPR screen data manager.

        Args:
            data_file (Path | DataFile | str): File with data (csv, tsv, or pkl).
            transformations (list[data_transformation] | None, optional): List of
            functions that take, mutate, and return a data frame.
            Defaults to None.
        """
        self._data: pd.DataFrame | None = None

        if transformations is None:
            self._transformations = []
        else:
            self._transformations = deepcopy(transformations)

        if isinstance(data_file, DataFile):
            self.data_file = data_path(data_file)
        elif isinstance(data_file, str):
            self.data_file = Path(data_file)
        elif isinstance(data_file, Path):
            self.data_file = data_file
        else:
            raise BaseException("Type not accepted for data file.")

        if not self.data_file.exists():
            raise DataFileDoesNotExist(self.data_file)
        if not self.data_file.is_file():
            raise DataFileIsNotAFile(self.data_file)
        if self.data_file.suffix not in SUPPORTED_DATA_FILES:
            raise UnsupportedDataFileType(self.data_file.suffix)

        return None

    # ---- Properties ----

    @property
    def data(self) -> pd.DataFrame:
        """Get the data from the manager."""
        return self.get_data()

    @property
    def data_is_loaded(self) -> bool:
        """Check if the data has been loaded."""
        return self._data is not None

    @property
    def transformations(self) -> list[data_transformation]:
        """Get the list of data transformations.

        Returns:
            list[data_transformation]: List of data transformations.
        """
        return deepcopy(self._transformations)

    @property
    def num_transformations(self) -> int:
        """Get the number of transformations.

        Returns:
            int: Number of transformations.
        """
        return len(self._transformations)

    # ---- Data dataframe ----

    def get_data(
        self,
        skip_transforms: bool = False,
        skip_validation: bool = False,
        force_reread: bool = False,
        read_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Get the data frame.

        If the data has already been loaded, it is returned without re-reading from
        file.

        Args:
            skip_transforms (bool, optional): Skip data transformations. Defaults to
            False.
            skip_validation (bool, optional): Skip data validation. Defaults to False.
            force_reread (bool, optional): Force the file to be read even if the data
            object already exists. Defaults to False.
            read_kwargs (dict[str, Any] | None, optional): Key-word arguments for the
            CSV-parsing function. Defaults to None.

        Raises:
            UnsupportedDataFileType: Data file type is not supported.

        Returns:
            pd.DataFrame: Requested data frame.
        """
        if self._data is not None and not force_reread:
            return self._data

        if self.data_file.suffix not in SUPPORTED_DATA_FILES:
            raise UnsupportedDataFileType(self.data_file.suffix)

        if read_kwargs is None:
            read_kwargs = {}
        if "dtype" not in read_kwargs:
            read_kwargs["dtype"] = {
                "p_dna_batch": str,
                "screen": str,
                "sgrna_target_chr": str,
                "rna_expr": np.float64,
            }

        if self.data_file.suffix == ".csv":
            self._data = pd.read_csv(self.data_file, **read_kwargs)
        elif self.data_file.suffix == ".tsv":
            self._data = pd.read_csv(self.data_file, sep="\t", **read_kwargs)
        elif self.data_file.suffix == ".pkl":
            self._data = pd.read_pickle(self.data_file)
        else:
            raise UnsupportedDataFileType(self.data_file.suffix)

        assert isinstance(self._data, pd.DataFrame)

        if not skip_transforms:
            self._data = self.apply_transformations(self._data)
        if not skip_validation:
            self._data = self.apply_validation(self._data)

        return self._data

    def set_data(self, data: pd.DataFrame, apply_transformations: bool = True) -> None:
        """Set the CRISPR screen data.

        Args:
            data (pd.DataFrame): New data (a copy is made).
            apply_transformations (bool, optional): Should the transformations be
              applied? Defaults to True.
        """
        self._data = data.copy()
        if apply_transformations:
            self._data = self.apply_transformations(self._data)
        self._data = self.apply_validation(self._data)
        return None

    def clear_data(self) -> None:
        """Clear the CRISPR screen data."""
        self._data = None

    # ---- Transformations ----

    def _apply_default_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        return set_achilles_categorical_columns(
            data, ordered=True, sort_cats=True
        ).pipe(set_chromosome_categories, "sgrna_target_chr")

    def add_transformation(
        self, fxn: data_transformation | list[data_transformation]
    ) -> None:
        """Add a new transformation.

        The new transformation is added to the end of the current list.

        Args:
            fxn (data_transformation | list[data_transformation]: Data transforming
            function(s).

        Returns:
            None
        """
        if isinstance(fxn, list):
            self._transformations += fxn
        else:
            self._transformations.append(fxn)
        return None

    def insert_transformation(self, fxn: data_transformation, at: int) -> None:
        """Insert a new transformation at a specified index.

        Args:
            fxn (DataFrameTransformation): Data transforming function.
            at (int): Insertion index.

        Returns:
            None
        """
        self._transformations.insert(at, fxn)
        return None

    def clear_transformations(self) -> None:
        """Clear the list of transformations."""
        self._transformations = []

    def set_transformations(
        self, new_transformations: list[data_transformation]
    ) -> None:
        """Set the list of transformations.

        A deep copy of the list of transformations is make.

        Args:
            new_transformations (list[DataFrameTransformation]): New list of data
            transforming functions.
        """
        self._transformations = deepcopy(new_transformations)
        return None

    def apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformations to a dataframe.

        Args:
            data (pd.DataFrame): Dataframe to transform.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for fxn in self._transformations:
            data = fxn(data)
        data = self._apply_default_transformations(data)
        return data

    # ---- Validation ----

    @property
    def data_schema(self) -> DataFrameSchema:
        """Data validation schema.

        Returns:
            DataFrameSchema: Pandera data schema.
        """
        return DataFrameSchema(
            {
                "sgrna": Column("category"),
                "hugo_symbol": Column(
                    "category",
                    checks=[
                        # A sgRNA maps to a single gene ("hugo_symbol")
                        pa.Check(check_unique_groups, groupby="sgrna"),
                    ],
                ),
                "lineage": Column(
                    "category",
                    checks=[
                        # Each cell line maps to a single lineage.
                        pa.Check(check_unique_groups, groupby="depmap_id")
                    ],
                ),
                "depmap_id": Column("category"),
                "p_dna_batch": Column("category"),
                "sgrna_target_chr": Column("category"),
                "lfc": Column(
                    float,
                    checks=[check_finite(), check_between(-20, 20)],
                ),
                "counts_final": Column(
                    float,
                    checks=[
                        check_finite(nullable=True),
                        check_nonnegative(nullable=True),
                    ],
                    nullable=True,
                    coerce=True,
                ),
                "copy_number": Column(
                    float,
                    checks=[
                        check_finite(nullable=True),
                        check_nonnegative(nullable=True),
                    ],
                    nullable=True,
                ),
                "rna_expr": Column(
                    float,
                    checks=[
                        check_finite(nullable=True),
                        check_nonnegative(nullable=True),
                    ],
                    nullable=True,
                ),
                "num_mutations": Column(
                    int, checks=[check_nonnegative(), check_finite()]
                ),
            }
        )

    def apply_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data validation to a dataframe.

        Args:
            data (pd.DataFrame): Dataframe to validate.

        Returns:
            pd.DataFrame: Validated dataframe.
        """
        return self.data_schema.validate(data)


LineageSubtypeGeneMap = dict[str, dict[str, set[str]]]
LineageGeneMap = dict[str, set[str]]


class CancerGeneDataManager:
    """Manage cancer gene data."""

    # TODO: change the API to make it easier to "reduce".
    # TODO: add other API features for filter lineages, sublineages, etc.

    def __init__(self) -> None:
        """Create a cancer gene data manager."""
        return None

    def _read_cancer_gene_dict(self, json_path: Path) -> LineageSubtypeGeneMap:
        with open(json_path) as json_file:
            lineage_genes = json.load(json_file)

        for sublineage_genes in lineage_genes.values():
            for sublineage, genes in sublineage_genes.items():
                if isinstance(genes, str):
                    sublineage_genes[sublineage] = {genes}
                elif isinstance(genes, list):
                    sublineage_genes[sublineage] = set(genes)
        return lineage_genes

    def bailey_2018_cancer_genes(self) -> LineageSubtypeGeneMap:
        """The cancer genes from Bailey et al., Cell, 2018.

        Returns:
            CancerGeneMap: Map of cancer types to their associated genes.
        """
        return self._read_cancer_gene_dict(
            modeling_data_dir() / "bailey-cancer-genes-dict.json"
        )

    def cosmic_cancer_genes(self) -> LineageSubtypeGeneMap:
        """The cancer genes from COSMIC.

        Returns:
            CancerGeneMap: Map of cancer types to their associated genes.
        """
        return self._read_cancer_gene_dict(
            modeling_data_dir() / "cgc-cancer-genes-dict.json"
        )

    def reduce_to_lineage(self, gene_map: LineageSubtypeGeneMap) -> LineageGeneMap:
        """Reduce a lineage-subtype-specific gene map to lineage-specific.

        Args:
            gene_map (LineageSubtypeGeneMap): Lineage-subtype-specific gene map.

        Returns:
            LineageGeneMap: Lineage-specific gene map.
        """
        return {
            line: merge_sets(sublines.values()) for line, sublines in gene_map.items()
        }
