"""Custom exceptions for the project."""

from pathlib import Path


class UnsupportedFitMethod(Exception):
    """The indicated fit method is not supported."""

    pass


class IncorrectNumberOfFilesFoundError(Exception):
    """Incorrect number of files found."""

    def __init__(self, expected: int, found: int) -> None:
        """Create an IncorrectNumberOfFilesFoundError instance.

        Args:
            expected (int): Expected number of files.
            found (int): Number of files found.
        """
        self.expected = expected
        self.found = found
        self.message = f"Expected {self.expected} files, but found {self.found} files."
        super().__init__(self.message)


class UnsupportedFileTypeError(Exception):
    """Unsupported file type."""

    def __init__(self, suffix: str) -> None:
        """Create an UnsupportedFileTypeError instance.

        Args:
            suffix (str): The unsupported suffix that some monster tried to use.
        """
        self.suffix = suffix
        self.message = f"File type '{self.suffix}' is not supported."
        super().__init__(self.message)


class NoDirectorySpecified(Exception):
    """No directory is specified when one is required."""

    pass


class ResourceRequestUnkown(NotImplementedError):
    """Exception raised when a resource request cannot be fullfilled."""

    def __init__(self, resource: str, id: str) -> None:
        """Create a ResourceRequestUnkown instance.

        Args:
            resource (str): Resource being requested.
            id (str): Some name to help identify the source of the problem.
        """
        self.resource = resource
        self.id = id
        self.message = f"Unknown {self.resource} for '{self.id}'"
        super().__init__(self.message)


class CacheDoesNotExistError(FileNotFoundError):
    """Cache does not exist."""

    def __init__(self, dir: Path | str) -> None:
        """Create a CacheDoesNotExistError error.

        Args:
            dir (Union[Path, str]): Expected location of cached data.
        """
        self.dir = dir
        self.message = str(dir)
        super().__init__(self.message)


class ShapeError(BaseException):
    """Shape error."""

    def __init__(
        self,
        expected_shape: int | tuple[int, ...],
        actual_shape: int | tuple[int, ...],
    ) -> None:
        """Create a ShapeError error.

        Args:
            expected_shape (Union[int, tuple[int, ...]]): Expected shape.
            actual_shape (Union[int, tuple[int, ...]]): Actual (observed) shape.
        """
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.message = (
            f"Expected shape {self.expected_shape} - actual shape {self.actual_shape}."
        )
        super().__init__(self.message)


class RequiredArgumentError(BaseException):
    """Errors concerning required arguments that are not language-enforced."""

    pass


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


class ConfigurationNotFound(BaseException):
    """Configuration not found in a configuration file."""

    ...
