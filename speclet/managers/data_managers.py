"""Data management classes."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Final, Optional, Union

import pandas as pd
from pandera import Column, DataFrameSchema

from speclet.data_processing.crispr import set_achilles_categorical_columns
from speclet.data_processing.validation import (
    check_between,
    check_finite,
    check_nonnegative,
)
from speclet.io import DataFile, data_path

data_transformation = Callable[[pd.DataFrame], pd.DataFrame]


class DataNotLoadedException(BaseException):
    """Data not loaded exception."""

    pass


class DataFileDoesNotExist(BaseException):
    """Data file does not exist."""

    def __init__(self, file: Path) -> None:
        """Data file does not exist."""
        msg = f"Data file '{file}' not found."
        super().__init__(msg)
        return None


class DataFileIsNotAFile(BaseException):
    """Data file is not a file."""

    def __init__(self, file: Path) -> None:
        """Data file is not a file."""
        msg = f"Path must be to a file: '{file}'."
        super().__init__(msg)
        return None


class UnsupportedDataFileType(BaseException):
    """Unsupported data file type."""

    def __init__(self, suffix: str) -> None:
        """Unsupported data file type."""
        msg = f"File type '{suffix}' is not supported."
        super().__init__(msg)
        return None


class ColumnsNotUnique(BaseException):
    """Column names are not unique."""

    def __init__(self) -> None:
        """Columns not unique."""
        msg = "Column names must be unique."
        super().__init__(msg)
        return None


class CrisprScreenDataManager:
    """Manage CRISPR screen data."""

    data_file: Path
    _data: Optional[pd.DataFrame]
    _transformations: list[data_transformation]

    _supported_filetypes: Final[set[str]] = {".csv", ".tsv", ".pkl"}

    def __init__(
        self,
        data_file: Union[Path, DataFile, str],
        transformations: Optional[list[data_transformation]] = None,
    ) -> None:
        """Create a CRISPR screen data manager.

        Args:
            data_file (Union[Path, DataFile, str]): File with data (csv, tsv, or pkl).
            transformations (Optional[list[DataFrameTransformation]], optional): List of
            functions that take, mutate, and return a data frame.
            Defaults to None.
        """
        self._data = None

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
        if self.data_file.suffix not in self._supported_filetypes:
            raise UnsupportedDataFileType(self.data_file.suffix)

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
        force_reread: bool = False,
        read_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Get the dataframe.

        If the data has already been loaded, it is returned without re-reading from
        file.

        Args:
            skip_transforms (bool, optional): Skip data transformations. Defaults to
            False.
            force_reread (bool, optional): Force the file to be read even if the data
            object already exists. Defaults to False.
            read_kwargs (Optional[dict[str, Any]], optional): Key-word arguments for the
            CSV-parsing function. Defaults to None.

        Raises:
            UnsupportedDataFileType: Data file type is not supported.

        Returns:
            pd.DataFrame: Requested dataframe.
        """
        if self._data is not None and not force_reread:
            return self._data

        if self.data_file.suffix not in self._supported_filetypes:
            raise UnsupportedDataFileType(self.data_file.suffix)

        if read_kwargs is None:
            read_kwargs = {}

        if self.data_file.suffix == ".csv":
            self._data = pd.read_csv(self.data_file)
        elif self.data_file.suffix == ".tsv":
            self._data = pd.read_csv(self.data_file, sep="\t")
        elif self.data_file.suffix == ".pkl":
            self._data = pd.read_pickle(self.data_file)
        else:
            raise UnsupportedDataFileType(self.data_file.suffix)

        assert isinstance(self._data, pd.DataFrame)

        if not skip_transforms:
            self._data = self.apply_transformations(self._data)

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
        return set_achilles_categorical_columns(data)

    def add_transformation(
        self, fxn: Union[data_transformation, list[data_transformation]]
    ) -> None:
        """Add a new transformation.

        The new transformation is added to the end of the current list.

        Args:
            fxn (Union[DataFrameTransformation, list[DataFrameTransformation]]): Data
              transforming function(s).

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
        data = self._apply_default_transformations(data)
        for fxn in self._transformations:
            data = fxn(data)
        return data

    # ---- Validation ----

    @property
    def data_schema(self) -> DataFrameSchema:
        """Data validation schema.

        # TODO: check for unique pairing of sgRNA and gene.
        # TODO: check for unique pairing of cell line and lineage.
        # TODO: add tests for both of these checks

        Returns:
            DataFrameSchema: Pandera data schema.
        """
        return DataFrameSchema(
            {
                "sgrna": Column("category"),
                "hugo_symbol": Column("category"),
                "lineage": Column("category"),
                "depmap_id": Column("category"),
                "p_dna_batch": Column("category"),
                "sgrna_target_chr": Column("category"),
                "lfc": Column(
                    float,
                    checks=[check_finite(), check_between(-20, 20)],
                ),
                "counts_final": Column(
                    int, checks=[check_finite(), check_nonnegative()]
                ),
                "copy_number": Column(
                    float, checks=[check_finite(), check_nonnegative()]
                ),
                "rna_expr": Column(float, checks=[check_finite(), check_nonnegative()]),
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
