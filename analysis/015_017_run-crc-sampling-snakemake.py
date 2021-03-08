#!/usr/bin/env python3

from pathlib import Path

import pretty_errors

PYMC3_MODEL_CACHE_DIR = "analysis/pymc3_model_cache/"
REPORTS_DIR = "reports/"

model_names = {
    "crc-m1": "CRC_model1",
}

rule all:
    input:
        expand(
            REPORTS_DIR + "{model}.md",
            model=list(model_names.keys()),
        )

rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "{model}/{model}.txt"
    params:
        model_name = lambda w: model_names[w.model]
    conda:
        "../environment.yml"
    shell:
        'python3 analysis/sampling_pymc3_models.py --model "{wildcards.model}" --name "{params.model_name}" --debug --touch'

rule report:
    input:
        model = PYMC3_MODEL_CACHE_DIR + "{model}/{model}.txt"
    output:
         ouput_notebook = REPORTS_DIR + "{model}.md"
    conda:
        "../environment.yml"
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + REPORTS_DIR  + "{wildcards.model}.ipynb && "
        "jupyter nbconvert --to markdown " + REPORTS_DIR + "{wildcards.model}.ipynb"
