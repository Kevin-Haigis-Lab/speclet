"""Manager for the posterior summaries saved of fit models."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PosteriorSummaryFiles:
    """Posterior summary files."""

    description: Path
    posterior_summary: Path
    posterior_predictions: Path


class PosteriorSummaryManager:
    """Posterior summary manager."""

    def __init__(self, id: str, cache_dir: Path | str) -> None:
        """Posterior summary manager.

        Args:
            id (str): Model ID.
            cache_dir (Path | str): Model caching directory.
        """
        self.id = id
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        assert self.cache_dir.exists(), "Cache directory does not exist."
        assert self.cache_dir.is_dir(), "Cache directory is not a directory."
        self.check_files_exist()

    def files(self) -> PosteriorSummaryFiles:
        """Get the summary files."""
        d = self.cache_dir / self.id
        return PosteriorSummaryFiles(
            description=d / "description.txt",
            posterior_summary=d / "posterior-summary.csv",
            posterior_predictions=d / "posterior-predictions.csv",
        )

    def check_files_exist(self) -> None:
        """Do all summary files exist."""
        for file in asdict(self.files()).values():
            if not file.exists():
                raise FileNotFoundError(file)

    def read_posterior_summary(self) -> pd.DataFrame:
        """Read the posterior summary file."""
        return pd.read_csv(self.files().posterior_summary)

    def read_posterior_predictions(self) -> pd.DataFrame:
        """Read the posterior predictions file."""
        return pd.read_csv(self.files().posterior_predictions)

    def read_description(self) -> str:
        """Read the description file."""
        files = self.files()
        with open(files.description, "r") as file:
            return "".join([line for line in file])
