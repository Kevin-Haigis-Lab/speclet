"""Parsing complex structures for use in Snakemake pipelines."""

from pathlib import Path

from pydantic import BaseModel, validate_arguments

from speclet.bayesian_models import BayesianModel
from speclet.pipelines import slurm_resources as slurm


class ParsedPipelineInformation(BaseModel):
    """Parsed pipeline information."""

    model_names: list[str]
    fit_methods: list[str]

    def __len__(self) -> int:
        """Number of models to fit."""
        assert len(self.model_names) == len(self.fit_methods)
        return len(self.model_names)


@validate_arguments
def get_models_names_fit_methods(
    config_path: Path, *, model_type: set[BayesianModel] | BayesianModel | None = None
) -> ParsedPipelineInformation:
    """Get model names and fit methods for ease-of-use in snakemake workflow.

    Args:
        config_path (Path): Path to a configuration file.
        model_type (BayesianModel | None, optional): Filter for a specific model or
        group of models.

    Returns:
        ParsedPipelineInformation: The information in a useful format for use in a
        snakemake workflow.
    """
    if isinstance(model_type, BayesianModel):
        model_type = {model_type}
    configs = slurm.read_resource_configs(config_path)
    names: list[str] = []
    fit_methods: list[str] = []
    for config in configs:
        if not config.active:
            continue
        if model_type is not None and config.model not in model_type:
            continue
        fms = list(config.slurm_resources.keys())
        names += [config.name] * len(fms)
        fit_methods += [fm.value for fm in fms]
    return ParsedPipelineInformation(model_names=names, fit_methods=fit_methods)
