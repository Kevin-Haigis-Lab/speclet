#!/usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Any, List

import papermill
from snakemake.io import Wildcards

from src.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from src.pipelines.snakemake_parsing_helpers import get_models_names_fit_methods
from src.pipelines.theano_flags import get_theano_flags
from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline
from src import project_config

# Global parameters.
N_CHAINS = 4
DEBUG = project_config.debug_status()

# Directory and file paths.
SCRATCH_DIR = "/n/scratch3/users/j/jc604"
SCRATCH_TEMP_DIR = f"{SCRATCH_DIR}/speclet/fitting-mcmc/"
PYMC3_MODEL_CACHE_DIR = "models/"
REPORTS_DIR = "reports/crc_model_sampling_reports/"
ENVIRONMENT_YAML = Path("default_environment.yaml").as_posix()

# Benchmarking jobs.
BENCHMARK_DIR = Path("benchmarks", "010_010_run-crc-sampling-snakemake")
if not BENCHMARK_DIR.exists():
    BENCHMARK_DIR.mkdir(parents=True)


#### ---- Model configurations ---- ####

MODEL_CONFIG = Path("models", "model-configs.yaml")
model_configuration_lists = get_models_names_fit_methods(
    MODEL_CONFIG, pipeline=SpecletPipeline.FITTING
)


if len(model_configuration_lists) == 0:
    raise BaseException("No models to run in pipeline in the configuration file.")


#### ---- Wildcard constrains ---- ####


wildcard_constraints:
    model_name="|".join(set(model_configuration_lists.model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    chain="\d+",


#### ---- Helpers ---- ####


def create_resource_manager(w: Wildcards, fit_method: ModelFitMethod) -> RM:
    return RM(
        name=w.model_name, fit_method=fit_method, config_path=MODEL_CONFIG, debug=DEBUG
    )


#### ---- Rules ---- ####


localrules:
    all,
    papermill_report,


rule all:
    input:
        expand(
            REPORTS_DIR + "{model_name}_{fit_method}.md",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


rule sample_mcmc:
    output:
        touch_file=PYMC3_MODEL_CACHE_DIR + "_{model_name}_chain{chain}_MCMC.txt",
        chain_dir=directory(SCRATCH_TEMP_DIR + "{model_name}_chain{chain}"),
    params:
        mem=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).memory,
        time=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).time,
        partition=lambda w: create_resource_manager(w, ModelFitMethod.MCMC).partition,
        config_file=MODEL_CONFIG.as_posix(),
    conda:
        ENVIRONMENT_YAML
    benchmark:
        BENCHMARK_DIR / "sample_mcmc/{model_name}_chain{chain}.tsv"
    priority: 20
    shell:
        get_theano_flags("{wildcards.model_name}_{wildcards.chain}_mcmc") + " "
        "src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  MCMC"
        "  {output.chain_dir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --random-seed {wildcards.chain}"
        "  --touch {output.touch_file}"


rule combine_mcmc:
    input:
        chains=expand(
            PYMC3_MODEL_CACHE_DIR + "_{{model_name}}_chain{chain}_MCMC.txt",
            chain=list(range(N_CHAINS)),
        ),
    output:
        touch_file=PYMC3_MODEL_CACHE_DIR + "_{model_name}_MCMC.txt",
    params:
        config_file=MODEL_CONFIG.as_posix(),
        combined_cache_dir=PYMC3_MODEL_CACHE_DIR,
        chain_dirs=directory(
            expand(
                SCRATCH_TEMP_DIR + "{{model_name}}_chain{chain}",
                chain=list(range(N_CHAINS)),
            )
        ),
    conda:
        ENVIRONMENT_YAML
    shell:
        get_theano_flags("{wildcards.model_name}_combine-mcmc") + " "
        "src/command_line_interfaces/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  {params.config_file}"
        "  {params.chain_dirs}"
        "  {params.combined_cache_dir}"
        "  --touch {output.touch_file}"


rule sample_advi:
    output:
        touch_file=PYMC3_MODEL_CACHE_DIR + "_{model_name}_ADVI.txt",
    params:
        mem=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).memory,
        time=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).time,
        partition=lambda w: create_resource_manager(w, ModelFitMethod.ADVI).partition,
        config_file=MODEL_CONFIG.as_posix(),
        cache_dir=PYMC3_MODEL_CACHE_DIR,
    conda:
        ENVIRONMENT_YAML
    benchmark:
        BENCHMARK_DIR / "sample_advi/{model_name}.tsv"
    priority: 10
    shell:
        get_theano_flags("{wildcards.model_name}_advi") + " "
        "src/command_line_interfaces/sampling_pymc3_models_cli.py"
        '  "{wildcards.model_name}"'
        " {params.config_file}"
        "  ADVI"
        " {params.cache_dir}"
        "  --mcmc-cores 1"
        "  --random-seed 7414"
        "  --touch {output.touch_file}"


rule papermill_report:
    input:
        template_nb=REPORTS_DIR + "model-report-template.ipynb",
    output:
        notebook=REPORTS_DIR + "{model_name}_{fit_method}.ipynb",
    version:
        "1.0"
    run:
        papermill.execute_notebook(
            input.template_nb,
            output.notebook,
            parameters={
                "CONFIG_PATH": MODEL_CONFIG.as_posix(),
                "MODEL_NAME": wildcards.model_name,
                "FIT_METHOD": wildcards.fit_method,
                "ROOT_CACHE_DIR": PYMC3_MODEL_CACHE_DIR,
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        model_touch=PYMC3_MODEL_CACHE_DIR + "_{model_name}_{fit_method}.txt",
        notebook=rules.papermill_report.output.notebook,
    output:
        markdown=REPORTS_DIR + "{model_name}_{fit_method}.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.notebook} && "
        "nbqa isort {input.notebook} --nbqa-mutate && "
        "nbqa black {input.notebook} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.notebook}"


BENCHMARK_REPORT = "reports/benchmarks.ipynb"
run_benchmark_nb_cmd = f"""
    jupyter nbconvert --to notebook --inplace --execute '{BENCHMARK_REPORT}' &&
    jupyter nbconvert --to markdown '{BENCHMARK_REPORT}'
"""


onsuccess:
    shell(run_benchmark_nb_cmd)


onerror:
    shell(run_benchmark_nb_cmd)
