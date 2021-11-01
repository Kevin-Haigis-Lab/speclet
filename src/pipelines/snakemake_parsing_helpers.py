"""Parsing complex structures for use in Snakemake pipelines."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, validate_arguments

import src.model_configuration as model_config
from src.model_configuration import ModelConfig
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline, assert_never


class ParsedPipelineInformation(BaseModel):
    """Parsed pipeline information."""

    models: list[str]
    model_names: list[str]
    fit_methods: list[str]

    def __len__(self) -> int:
        return len(self.models)


class _PipelineIntermediateInformation(BaseModel):
    """An intermediate step in the parsing process."""

    model: ModelOption
    model_name: str
    fit_method: ModelFitMethod


def _get_pipeline_fit_methods(
    config: ModelConfig, pipeline: SpecletPipeline
) -> Optional[list[ModelFitMethod]]:
    return config.pipelines.get(pipeline, None)


@validate_arguments
def get_models_names_fit_methods(
    config_path: Path, pipeline: SpecletPipeline
) -> ParsedPipelineInformation:
    """Get model names and fit methods for ease-of-use in snakemake workflow.

    Args:
        config_path (Path): Path to a configuration file.
        pipeline (SpecletPipeline): Name of the pipeline.

    Returns:
        ParsedPipelineInformation: The information in a useful format for use in a
        snakemake workflow.
    """
    model_configurations = model_config.read_model_configurations(config_path)
    model_config.check_model_names_are_unique(model_configurations)

    pipeline_informations: list[_PipelineIntermediateInformation] = []

    for config in model_configurations.configurations:
        fit_methods = _get_pipeline_fit_methods(config, pipeline)
        if fit_methods is None:
            continue
        for fit_method in fit_methods:
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
        dict[str, int]: A dictionary where the key is the model names and the value is
        the hash of the JSON serialization of the model's configuration.
    """
    model_configurations = model_config.read_model_configurations(config_path)
    hashes: dict[str, int] = {}
    for config in model_configurations.configurations:
        config.description = ""
        hashes[config.name] = hash(config.json().strip())
    return hashes
