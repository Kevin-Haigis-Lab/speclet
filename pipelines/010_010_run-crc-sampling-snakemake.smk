#!/usr/bin/env python3

from pathlib import Path

import papermill
import pretty_errors

PYMC3_MODEL_CACHE_DIR = "models/model_cache/pymc3_model_cache/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"

ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()

model_names = (
    ("crc_ceres_mimic", "CERES-base"),
    ("crc_ceres_mimic", "CERES-copynumber"),
    ("crc_ceres_mimic", "CERES-sgrnaint"),
    ("crc_ceres_mimic", "CERES-copynumber-sgrnaint"),
    ("speclet_one", "SpecletOne"),
    ("speclet_two", "SpecletTwo"),
)

models = [m for m, _ in model_names]
model_names = [n for _, n in model_names]


rule all:
    input:
        expand(
            REPORTS_DIR + "{model}_{model_name}.md",
            zip,
            model=models,
            model_name=model_names,
        ),


rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}.txt",
    params:
        debug=lambda w: "debug" if "debug" in w.model_name else "no-debug",
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py "
        '  "{wildcards.model}" '
        '  "{wildcards.model_name}" '
        "  --random-seed 7414 "
        "  --{params.debug}"
        "  --touch"


def is_debug(name: str) -> bool:
    return "debug" in name


rule papermill_report:
    input:
        model_touch=PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}.txt",
    output:
        notebook=REPORTS_DIR + "{model}_{model_name}.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "model-report-template.ipynb",
            output.notebook,
            parameters={
                "MODEL": wildcards.model,
                "MODEL_NAME": wildcards.model_name,
                "DEBUG": is_debug(wildcards.model_name),
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        notebook=REPORTS_DIR + "{model}_{model_name}.ipynb",
    output:
        markdown=REPORTS_DIR + "{model}_{model_name}.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
