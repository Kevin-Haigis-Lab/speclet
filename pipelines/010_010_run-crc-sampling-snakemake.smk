#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import papermill
import pretty_errors
from pydantic import BaseModel

import run_sampling_utils as utils

PYMC3_MODEL_CACHE_DIR = "models/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"
ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()

N_CHAINS = 4

#### ---- Models ---- ####


class ModelOption(str, Enum):
    """Model options."""

    speclet_test_model = "speclet-test-model"
    crc_ceres_mimic = "crc-ceres-mimic"
    speclet_one = "speclet-one"
    speclet_two = "speclet-two"
    speclet_three = "speclet-three"
    speclet_four = "speclet-four"


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"


class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    model: ModelOption
    fit_method: ModelFitMethod = ModelFitMethod.advi


models_configurations = []
# models_configurations += [
#     ModelConfig(name="SpecletTest-debug", model="speclet-test-model", fit_method="ADVI"),
#     ModelConfig(name="SpecletTest-debug", model="speclet-test-model", fit_method="MCMC"),
# ]
models_configurations += [
    ModelConfig(name="SpecletTwo-debug", model="speclet-two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-debug", model="speclet-two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet-two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet-two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo", model="speclet-two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo", model="speclet-two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras", model="speclet-two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras", model="speclet-two", fit_method="MCMC"),
]
models_configurations += [
    ModelConfig(name="SpecletThree-debug", model="speclet-three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree-debug", model="speclet-three", fit_method="MCMC"),
    ModelConfig(name="SpecletThree-kras-debug", model="speclet-three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree-kras-debug", model="speclet-three", fit_method="MCMC"),
    ModelConfig(name="SpecletThree", model="speclet-three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree", model="speclet-three", fit_method="MCMC"),
    ModelConfig(name="SpecletThree-kras", model="speclet-three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree-kras", model="speclet-three", fit_method="MCMC"),
]
models_configurations += [
    ModelConfig(name="SpecletFour-debug", model="speclet-four", fit_method="MCMC"),
    ModelConfig(name="SpecletFour", model="speclet-four", fit_method="MCMC"),
]

# Separate information in model configuration for `all` step to create wildcards.
models = [m.model.value for m in models_configurations]
model_names = [m.name for m in models_configurations]
fit_methods = [m.fit_method.value for m in models_configurations]


#### ---- Wildcard constrains ---- ####

wildcard_constraints:
    model="|".join([a.value for a in ModelOption]),
    model_name="|".join(set(model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    chain="\d+",

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
        debug=utils.cli_is_debug,
        mem=lambda w: utils.get_sample_models_memory(w, "MCMC"),
        time=lambda w: utils.get_sample_models_time(w, "MCMC"),
        partition=lambda w: utils.get_sample_models_partition(w, "MCMC"),
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
        debug=utils.cli_is_debug,
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
        debug=utils.cli_is_debug,
        mem=lambda w: utils.get_sample_models_memory(w, "MCMC"),
        time=lambda w: utils.get_sample_models_time(w, "MCMC"),
        partition=lambda w: utils.get_sample_models_partition(w, "MCMC"),
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
