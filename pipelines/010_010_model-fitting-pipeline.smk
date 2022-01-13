"""Model fitting pipeline."""

from enum import Enum
from pathlib import Path
from typing import Any, List

import papermill
from snakemake.io import Wildcards

from speclet.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from speclet.pipelines.snakemake_parsing_helpers import get_models_names_fit_methods
from speclet.pipelines.theano_flags import get_theano_flags
from speclet.project_enums import ModelFitMethod, SpecletPipeline
from speclet.project_configuration import fitting_pipeline_config


pipeline_config = fitting_pipeline_config()

# Global parameters.
N_CHAINS = pipeline_config.num_chains
DEBUG = pipeline_config.debug

# Directory and file paths.
TEMP_DIR = pipeline_config.temp_dir  # TODO: on 02, symlink to Scratch.
MODEL_CACHE_DIR = pipeline_config.model_cache_dir
REPORTS_DIR = pipeline_config.reports_dir
ENVIRONMENT_YAML = str(pipeline_config.env_yaml)

# Benchmarking jobs.
BENCHMARK_DIR = pipeline_config.benchmark_dir
if not BENCHMARK_DIR.exists():
    BENCHMARK_DIR.mkdir(parents=True)


#### ---- Model configurations ---- ####

MODEL_CONFIG = pipeline_config.models_config
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


def get_memory(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).memory


def get_time(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).memory


def get_partition(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).memory


#### ---- Rules ---- ####


localrules:
    all,
    papermill_report,


rule all:
    input:
        expand(
            str(REPORTS_DIR / "{model_name}_{fit_method}.md"),
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


rule sample_pymc3_mcmc:
    output:
        idata_path=str(
            TEMP_DIR / "{model_name}_PYMC3_MCMC_chain{chain}" / "posterior.json"
        ),
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.PYMC3_MCMC),
        time=lambda w: get_time(w, ModelFitMethod.PYMC3_MCMC),
        partition=lambda w: get_partition(w, ModelFitMethod.PYMC3_MCMC),
        config_file=str(MODEL_CONFIG),
        tempdir=str(TEMP_DIR),
        cache_name=lambda w: f"{w.model_name}_PYMC3_MCMC_chain{w.chain}",
    conda:
        ENVIRONMENT_YAML
    benchmark:
        str(BENCHMARK_DIR / "sample_pymc3_mcmc/{model_name}_chain{chain}.tsv")
    priority: 20
    shell:
        get_theano_flags("{wildcards.model_name}_{wildcards.chain}_mcmc") + " "
        "speclet/command_line_interfaces/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC3_MCMC"
        "  {params.tempdir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --cache-name {params.cache_name}"


rule combine_mcmc:
    input:
        chains=expand(
            str(
                TEMP_DIR / "{{model_name}}_PYMC3_MCMC_chain{chain}" / "posterior.json"
            ),
            chain=list(range(N_CHAINS)),
        ),
    output:
        combined_chains=str(
            MODEL_CACHE_DIR / "{model_name}_PYMC3_MCMC" / "posterior.json"
        ),
    params:
        n_chains=N_CHAINS,
        combined_cache_dir=str(MODEL_CACHE_DIR),
        config_file=str(MODEL_CONFIG),
        cache_dir=str(TEMP_DIR),
    conda:
        ENVIRONMENT_YAML
    shell:
        get_theano_flags("{wildcards.model_name}_combine-mcmc") + " "
        "speclet/command_line_interfaces/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  PYMC3_MCMC"
        "  {params.n_chains}"
        "  {params.config_file}"
        "  {params.cache_dir}"
        "  {params.combined_cache_dir}"


rule sample_pymc3_advi:
    output:
        idata_path=str(MODEL_CACHE_DIR / "{model_name}_PYMC3_ADVI" / "posterior.json"),
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.PYMC3_ADVI),
        time=lambda w: get_time(w, ModelFitMethod.PYMC3_ADVI),
        partition=lambda w: get_partition(w, ModelFitMethod.PYMC3_ADVI),
        config_file=str(MODEL_CONFIG),
        cache_dir=str(MODEL_CACHE_DIR),
    conda:
        ENVIRONMENT_YAML
    benchmark:
        BENCHMARK_DIR / "sample_pymc3_advi/{model_name}.tsv"
    priority: 10
    shell:
        get_theano_flags("{wildcards.model_name}_advi") + " "
        "speclet/command_line_interfaces/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC3_ADVI"
        "  {params.cache_dir}"
        "  --mcmc-cores 1"


rule papermill_report:
    input:
        template_nb=str(REPORTS_DIR / "model-report-template.ipynb"),
    output:
        notebook=str(REPORTS_DIR / "{model_name}_{fit_method}.ipynb"),
    version:
        "1.0"
    run:
        papermill.execute_notebook(
            input.template_nb,
            output.notebook,
            parameters={
                "MODEL_NAME": wildcards.model_name,
                "FIT_METHOD_STR": wildcards.fit_method,
                "CONFIG_PATH": str(MODEL_CONFIG),
                "ROOT_CACHE_DIR": str(MODEL_CACHE_DIR),
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        idata_path=str(MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "posterior.json"),
        notebook=rules.papermill_report.output.notebook,
    output:
        markdown=str(REPORTS_DIR / "{model_name}_{fit_method}.md"),
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.notebook} && "
        "nbqa isort --profile=black {input.notebook} --nbqa-mutate && "
        "nbqa black {input.notebook} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.notebook}"


BENCHMARK_REPORT = "reports/benchmarks.ipynb"
run_benchmark_nb_cmd = f"""
    jupyter nbconvert --to notebook --inplace --execute '{BENCHMARK_REPORT}' &&
    nbqa isort --profile=black '{BENCHMARK_REPORT}' --nbqa-mutate &&
    nbqa black '{BENCHMARK_REPORT}' --nbqa-mutate &&
    jupyter nbconvert --to markdown '{BENCHMARK_REPORT}'
"""


onsuccess:
    shell(run_benchmark_nb_cmd)


onerror:
    shell(run_benchmark_nb_cmd)
