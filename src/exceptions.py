"""Custom exceptions for the project."""

# NOTE: Only import built in libraries so can use in any venv.


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
