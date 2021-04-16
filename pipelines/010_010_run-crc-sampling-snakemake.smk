#!/usr/bin/env python3

from pathlib import Path

import papermill
import pretty_errors

PYMC3_MODEL_CACHE_DIR = "models/model_cache/pymc3_model_cache/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"

ENVIRONMENT_YAML = Path("default_environment.yml").as_posix()

model_names = {
    "crc_m1": "CRC-model1",
    "crc_ceres-mimic": "CERES-base",
    "crc_ceres-mimic": "CERES-copynumber",
    "crc_ceres_mimic": "CERES-copynumber-sgrnaint",
}


rule all:
    input:
        expand(
            REPORTS_DIR + "{model}_{model_name}.md",
            zip,
            model=list(model_names.keys()),
            model_name=list(model_names.values()),
        ),


rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "{model_name}/{model}_{model_name}.txt",
    conda:
        ENVIRONMENT_YAML
    shell:
        "python3 src/command_line_interfaces/sampling_pymc3_models_cli.py "
        '  "{wildcards.model}" '
        '  "{wildcards.model_name}" '
        "  --debug "
        "  --random-seed 7414 "
        "  --touch"


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
                "DEBUG": True,
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
