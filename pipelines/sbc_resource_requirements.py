"""Manage resources for the simulation-based calibration pipeline."""


class SBCResourceManager:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def memory(self) -> str:
        if "mcmc" in self.name:
            return "3000"
        else:
            return "1600"

    @property
    def time(self) -> str:
        if "mcmc" in self.name:
            return "02:00:00"
        else:
            return "00:15:00"

    @property
    def cores(self) -> int:
        if "mcmc" in self.name:
            return 4
        else:
            return 1
