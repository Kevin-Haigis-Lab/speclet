#!/usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Any, List

import papermill

from src.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from src.pipelines.snakemake_parsing_helpers import get_models_names_fit_methods
from src.project_enums import ModelFitMethod, ModelOption

PYMC3_MODEL_CACHE_DIR = "models/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"
ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()

N_CHAINS = 4

#### ---- Model configurations ---- ####

MODEL_CONFIG = Path("models", "model-configs.yaml")

# Separate information in model configuration for `all` step to create wildcards.
model_configuration_lists = get_models_names_fit_methods(MODEL_CONFIG)

#### ---- Wildcard constrains ---- ####

wildcard_constraints:
    model="|".join([a.value for a in ModelOption]),
    model_name="|".join(set(model_configuration_lists.model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    chain="\d+",


#### ---- Helpers ---- ####

def create_resource_manager(w: Any, fit_method: ModelFitMethod) -> RM:
    return RM(model=w.model, name=w.model_name, fit_method=fit_method)

def cli_is_debug(w: Any) -> str:
    return create_resource_manager(w=w, fit_method=ModelFitMethod.ADVI).is_debug_cli()


#### ---- Rules ---- ####


rule all:
    input:
        expand(
            REPORTS_DIR + "{model}_{model_name}_{fit_method}.md",
            zip,
            model=model_configuration_lists.models,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


rule sample_mcmc:
    output:
        touch_file=PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_chain{chain}_MCMC.txt",
    params:
        mem=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).memory,
        time=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).time,
        partition=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).partition,
        debug=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).is_debug_cli(),
        config_file=MODEL_CONFIG.as_posix(),
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model}"'
        '  "{wildcards.model_name}_chain{wildcards.chain}"'
        " {params.config_file}"
        "  --fit-method MCMC"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --random-seed 7414"
        "  {params.debug}"
        "  --touch {output.touch_file}"

rule combine_mcmc:
    input:
        chains=expand(
            PYMC3_MODEL_CACHE_DIR + "_{{model}}_{{model_name}}_chain{chain}_MCMC.txt",
            chain=list(range(N_CHAINS))
        ),
    output:
        touch_file=PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_MCMC.txt",
    params:
        debug=cli_is_debug,
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/combine_mcmc_chains_cli.py"
        "  {wildcards.model}"
        "  {wildcards.model_name}"
        "  {input.chains}"
        "  --touch-file {output.touch_file}"
        "  {params.debug}"


rule sample_advi:
    output:
        PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_ADVI.txt",
    params:
        mem=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).memory,
        time=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).time,
        partition=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).partition,
        debug=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).is_debug_cli(),
        config_file=MODEL_CONFIG.as_posix(),
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model}"'
        '  "{wildcards.model_name}"'
        " {params.config_file}"
        "  --fit-method ADVI"
        "  --mcmc-cores 1"
        "  --random-seed 7414"
        "  {params.debug}"
        "  --touch"

rule papermill_report:
    input:
        model_touch=PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_{fit_method}.txt",
    output:
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "model-report-template.ipynb",
            output.notebook,
            parameters={
                "MODEL": wildcards.model,
                "MODEL_NAME": wildcards.model_name,
                "DEBUG": utils.is_debug(wildcards.model_name),
                "FIT_METHOD": wildcards.fit_method,
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}.ipynb",
    output:
        markdown=REPORTS_DIR + "{model}_{model_name}_{fit_method}.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.notebook} && "
        "nbqa black {input.notebook} --nbqa-mutate && "
        "nbqa isort {input.notebook} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.notebook}"
