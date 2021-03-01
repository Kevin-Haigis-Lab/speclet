#!/usr/bin/env python3

from pathlib import Path

import pretty_errors

PYMC3_MODEL_CACHE_DIR = "analysis/pymc3_model_cache"

model_names = {
    "crc-m1": "CRC_model1",
    "crc-m2": "CRC_model2",
    "crc-m3": "CRC_model3",
}

final_notebook = "analysis/015_020_crc-model-analysis"

rule all:
    input:
        final_notebook + ".md"

rule sample_models:
    output:
        PYMC3_MODEL_CACHE_DIR + "/{model}/{model}.txt"
    params:
        model_name = lambda w: model_names[w.model]
    conda:
        "../environment.yml"
    shell:
        'python3 analysis/sampling_pymc3_models.py --model "{wildcards.model}" --name "{params.model_name}" --debug --touch'

rule execute_analysis:
    input:
        models=expand(PYMC3_MODEL_CACHE_DIR + "/{model}/{model}.txt", model=list(model_names.keys()))
    output:
        final_notebook + ".md"
    conda:
        "../environment.yml"
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " +  final_notebook +  ".ipynb && "
        "jupyter nbconvert --to markdown "+ final_notebook + ".ipynb"
