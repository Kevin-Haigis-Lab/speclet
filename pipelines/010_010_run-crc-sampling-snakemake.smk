#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import papermill
import pretty_errors
from pydantic import BaseModel

PYMC3_MODEL_CACHE_DIR = "models/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"

ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()


class ModelOption(str, Enum):
    """Model options."""

    crc_ceres_mimic = "crc_ceres_mimic"
    speclet_one = "speclet_one"
    speclet_two = "speclet_two"


class ModelFitMethod(str, Enum):
    """Available fit methods."""

    advi = "ADVI"
    mcmc = "MCMC"

class ModelConfig(BaseModel):
    """Model configuration format."""

    name: str
    model: ModelOption
    fit_method: ModelFitMethod = ModelFitMethod.advi


models_configurations = [
    # ModelConfig(name="CERES-base-debug", model="crc_ceres_mimic", fit_method="ADVI"),
    # ModelConfig(name="CERES-base-debug", model="crc_ceres_mimic", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-debug", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras-debug", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo", model="speclet_two", fit_method="MCMC"),
    ModelConfig(name="SpecletTwo-kras", model="speclet_two", fit_method="ADVI"),
    ModelConfig(name="SpecletTwo-kras", model="speclet_two", fit_method="MCMC"),
]

# Separate information in model configuration for `all` step to create wildcards.
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
#   key: [model][debug][fit_method]
sample_models_memory_lookup = {
    "crc_ceres_mimic": {
        True:  {"ADVI": 15, "MCMC": 20},
        False: {"ADVI": 20, "MCMC": 40}
    },
    "speclet_two": {
        True:  {"ADVI": 7, "MCMC": 30},
        False: {"ADVI": 30, "MCMC": 150}
    },
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
sample_models_time_lookup = {
    "crc_ceres_mimic": {
        True:  {"ADVI": "00:30:00", "MCMC": "00:30:00"},
        False: {"ADVI": "03:00:00", "MCMC": "06:00:00"},
    },
    "speclet_two": {
        True:  {"ADVI": "00:30:00", "MCMC": "12:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
}


def is_debug(name: str) -> bool:
    """Determine the debug status of model name."""
    return "debug" in name


def get_from_lookup(w, lookup_dict):
    """Generic dictionary lookup for the params in the `sample_models` step."""
    return lookup_dict[w.model][is_debug(w.model_name)][w.fit_method]


def get_sample_models_memory(w) -> int:
    """Memory required for the `sample_models` step."""
    try:
        return get_from_lookup(w, sample_models_memory_lookup) * 1000

    except:
        if is_debug(w.model_name):
            return 7 * 1000
        else:
            return 20 * 1000


def get_sample_models_time(w) -> str:
    """Time required for the `sample_models` step."""
    try:
        return get_from_lookup(w, sample_models_time_lookup)
    except:
        if is_debug(w.model_name):
            return "00:30:00"
        else:
            return "01:00:00"


def get_sample_models_partition(w) -> str:
    t = [int(x) for x in get_sample_models_time(w).split(":")]
    total_minutes = (t[0] * 60) + t[1]
    if total_minutes < (12 * 60):
        return "short"
    elif total_minutes < (5 * 24 * 60):
        return "medium"
    else:
        return "long"


rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}_{fit_method}.txt",
    params:
        cores=lambda w: "4" if w.fit_method == "MCMC" else "1",
        debug=lambda w: "debug" if "debug" in w.model_name else "no-debug",
        mem=get_sample_models_memory,
        time=get_sample_models_time,
        partition=get_sample_models_partition,
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
        "nbqa black {input.notebook} --nbqa-mutate && "
        "nbqa isort {input.notebook} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.notebook}"
