"""Parsing complex structures for use in Snakemake pipelines."""

from pathlib import Path
from typing import List

from pydantic import BaseModel, validate_arguments

from src.io import model_config
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline


class ParsedPipelineInformation(BaseModel):
    """Parsed pipeline information."""

    models: List[str]
    model_names: List[str]
    fit_methods: List[str]


class _PipelineIntermediateInformation(BaseModel):
    """An intermediate step in the parsing process."""

    model: ModelOption
    model_name: str
    fit_method: ModelFitMethod


@validate_arguments
def get_models_names_fit_methods(
    config_path: Path, pipeline: SpecletPipeline
) -> ParsedPipelineInformation:
    model_configurations = model_config.get_model_configurations(config_path)
    model_config.check_model_names_are_unique(model_configurations)

    pipeline_informations: List[_PipelineIntermediateInformation] = []

    for config in model_configurations.configurations:
        if pipeline not in config.pipelines:
            continue
        for fit_method in config.fit_methods:
            pipeline_informations.append(
                _PipelineIntermediateInformation(
                    model=config.model, model_name=config.name, fit_method=fit_method
                )
            )

    return ParsedPipelineInformation(
        models=[pi.model.value for pi in pipeline_informations],
        model_names=[pi.model_name for pi in pipeline_informations],
        fit_methods=[pi.fit_method.value for pi in pipeline_informations],
    )


def get_model_config_hashes(config_path: Path) -> dict[str, int]:
    """Create a dictionary of hashes for each model's configuration.

    The description is removed before serializing.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        dict[str, int]: A dictionary where the key is the model names and the value is the hash of the JSON serialization of the model's configuration.
    """
    model_configurations = model_config.get_model_configurations(config_path)
    hashes: dict[str, int] = {}
    for config in model_configurations.configurations:
        config.description = ""
        hashes[config.name] = hash(config.json().strip())
    return hashes
