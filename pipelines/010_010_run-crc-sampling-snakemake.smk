#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import papermill
import pretty_errors
from pydantic import BaseModel
from src.command_line_interfaces.cli_helpers import ModelOption, ModelFitMethod

PYMC3_MODEL_CACHE_DIR = "models/model_cache/pymc3_model_cache/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"

ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()


class ModelConfig(BaseModel):
    name: str
    model: ModelOption
    fit_method: ModelFitMethod = ModelFitMethod.advi


models_configurations = [
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="MCMC"),
]

# model_names = (
#     ("crc_ceres_mimic", "CERES-base"),
#     ("crc_ceres_mimic", "CERES-copynumber"),
#     ("crc_ceres_mimic", "CERES-sgrnaint"),
#     ("crc_ceres_mimic", "CERES-copynumber-sgrnaint"),
#     ("speclet_one", "SpecletOne"),
#     ("speclet_two", "SpecletTwo"),
#     ("speclet_two", "SpecletTwo-debug", "MCMC"),
#     ("speclet_two", "SpecletTwo-debug", "ADVI"),
# )

models = [m.model.value for m in models_configurations]
model_names = [m.name for m in models_configurations]
fit_methods = [m.fit_method.value for m in models_configurations]


rule all:
    input:
        expand(
            REPORTS_DIR + "{model}_{model_name}_{fit_method}.md",
            zip,
            model=models,
            model_name=model_names,
            fit_method=fit_methods,
        ),


# RAM required for each configuration (in GB -> mult by 1000).
sample_models_memory_lookup = {
    "speclet_two": {True: {"ADVI": 5, "MCMC": 5}, False: {"ADVI": 20, "MCMC": 20}}
}

# Time required for each configuration.
sample_models_time_lookup = {
    "speclet_two": {
        True: {"ADVI": "00:12:00", "MCMC": "00:30:00"},
        False: {"ADVI": "06:00:00", "MCMC": "06:00:00"},
    }
}


def is_debug(name: str) -> bool:
    return "debug" in name


def get_from_lookup(w, lookup_dict) -> Any:
    return lookup_dict[w.model][is_debug(w.model_name)][w.fit_method]


def get_sample_models_memory(w):
    return get_from_lookup(w, sample_models_memory_lookup)


def get_sample_models_time(w):
    return get_from_lookup(w, sample_models_time_lookup)


rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}_{fit_method}.txt",
    params:
        cores=lambda w: "4" if w.fit_method == "MCMC" else "1",
        debug=lambda w: "debug" if "debug" in w.model_name else "no-debug",
        mem=get_sample_models_memory,
        time=get_sample_models_time,
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
        model_touch=(
            PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}_{fit_method}.txt"
        ),
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
        "nbqa black {input.notebook} && nbqa isort {input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
