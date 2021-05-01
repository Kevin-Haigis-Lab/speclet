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


#### ---- Models ---- ####


class ModelOption(str, Enum):
    """Model options."""

    crc_ceres_mimic = "crc_ceres_mimic"
    speclet_one = "speclet_one"
    speclet_two = "speclet_two"
    speclet_three = "speclet_three"
    speclet_four = "speclet_four"


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
models_configurations += [
    # ModelConfig(name="CERES-base-debug", model="crc_ceres_mimic", fit_method="ADVI"),
    # ModelConfig(name="CERES-base-debug", model="crc_ceres_mimic", fit_method="MCMC"),
]
models_configurations += [
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras", model="speclet_two", fit_method="MCMC"),
]
models_configurations += [
    ModelConfig(name="SpecletThree-debug", model="speclet_three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree-debug", model="speclet_three", fit_method="MCMC"),
    ModelConfig(
        name="SpecletThree-kras-debug", model="speclet_three", fit_method="ADVI"
    ),
    ModelConfig(
        name="SpecletThree-kras-debug", model="speclet_three", fit_method="MCMC"
    ),
    ModelConfig(name="SpecletThree", model="speclet_three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree-kras", model="speclet_three", fit_method="ADVI"),
    ModelConfig(name="SpecletThree", model="speclet_three", fit_method="MCMC"),
    ModelConfig(name="SpecletThree-kras", model="speclet_three", fit_method="MCMC"),
]
models_configurations += [
    # ModelConfig(name="SpecletFour-debug", model="speclet_four", fit_method="ADVI"),
    ModelConfig(name="SpecletFour-debug", model="speclet_four", fit_method="MCMC"),
    ModelConfig(name="SpecletFour", model="speclet_four", fit_method="MCMC"),
]

# Separate information in model configuration for `all` step to create wildcards.
models = [m.model.value for m in models_configurations]
model_names = [m.name for m in models_configurations]
fit_methods = [m.fit_method.value for m in models_configurations]


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


rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "_{model}_{model_name}_{fit_method}.txt",
    params:
        cores=lambda w: "4" if w.fit_method == "MCMC" else "1",
        debug=lambda w: "debug" if "debug" in w.model_name else "no-debug",
        mem=utils.get_sample_models_memory,
        time=utils.get_sample_models_time,
        partition=utils.get_sample_models_partition,
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py "
        '  "{wildcards.model}" '
        '  "{wildcards.model_name}" '
        "  --fit-method {wildcards.fit_method} "
        "  --mcmc-chains {params.cores} "
        "  --mcmc-cores {params.cores} "
        "  --random-seed 7414 "
        "  --{params.debug}"
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
                "DEBUG": is_debug(wildcards.model_name),
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
