#!/usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Any

import papermill

from src.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from src.pipelines.pipeline_classes import ModelOption, model_config_from_yaml
from src.project_enums import ModelFitMethod

PYMC3_MODEL_CACHE_DIR = "models/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"
ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()

N_CHAINS = 4

MODEL_CONFIG = Path("pipelines", "model-configurations.yaml")

#### ---- Model configurations ---- ####

model_configurations = model_config_from_yaml(MODEL_CONFIG).configurations

# Separate information in model configuration for `all` step to create wildcards.
models = [m.model.value for m in model_configurations]
model_names = [m.name for m in model_configurations]
fit_methods = [m.fit_method.value for m in model_configurations]


#### ---- Wildcard constrains ---- ####

wildcard_constraints:
    model="|".join([a.value for a in ModelOption]),
    model_name="|".join(set(model_names)),
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
            model=models,
            model_name=model_names,
            fit_method=fit_methods,
        ),

rule sample_mcmc:
    output:
        PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_chain{chain}_MCMC.txt",
    params:
        mem=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).memory,
        time=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).time,
        partition=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).partition,
        debug=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).is_debug_cli(),
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model}"'
        '  "{wildcards.model_name}_chain{wildcards.chain}"'
        "  --fit-method MCMC"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --random-seed 7414"
        "  {params.debug}"
        "  --touch"

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
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model}"'
        '  "{wildcards.model_name}"'
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
