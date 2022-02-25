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
from speclet.pipelines.aesara_flags import get_aesara_flags
from speclet.project_enums import ModelFitMethod, SpecletPipeline
from speclet.project_configuration import fitting_pipeline_config


pipeline_config = fitting_pipeline_config()

# Global parameters.
N_CHAINS = pipeline_config.num_chains
DEBUG = pipeline_config.debug

# Directory and file paths.
TEMP_DIR = pipeline_config.temp_dir
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
    return create_resource_manager(w=w, fit_method=fit_method).time


def get_partition(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).partition


#### ---- Rules ---- ####


localrules:
    all,
    papermill_report,


rule all:
    input:
        report=expand(
            str(REPORTS_DIR / "{model_name}_{fit_method}.md"),
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        description=expand(
            str(MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "description.txt"),
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        summary=expand(
            str(
                MODEL_CACHE_DIR
                / "{model_name}_{fit_method}"
                / "posterior-summary.csv",
            ),
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        post_pred=expand(
            str(
                MODEL_CACHE_DIR
                / "{model_name}_{fit_method}"
                / "posterior-predictions.csv",
            ),
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


# --- Stan MCMC ---


rule sample_stan_mcmc:
    output:
        idata_path=str(
            TEMP_DIR / "{model_name}_STAN_MCMC_chain{chain}" / "posterior.netcdf"
        ),
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.STAN_MCMC),
        time=lambda w: get_time(w, ModelFitMethod.STAN_MCMC),
        partition=lambda w: get_partition(w, ModelFitMethod.STAN_MCMC),
        config_file=str(MODEL_CONFIG),
        tempdir=str(TEMP_DIR),
        cache_name=lambda w: f"{w.model_name}_STAN_MCMC_chain{w.chain}",
    conda:
        ENVIRONMENT_YAML
    benchmark:
        str(BENCHMARK_DIR / "sample_stan_mcmc/{model_name}_chain{chain}.tsv")
    priority: 20
    shell:
        "speclet/command_line_interfaces/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  STAN_MCMC"
        "  {params.tempdir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --cache-name {params.cache_name}"


rule combine_stan_mcmc:
    input:
        chains=expand(
            str(
                TEMP_DIR / "{{model_name}}_STAN_MCMC_chain{chain}" / "posterior.netcdf"
            ),
            chain=list(range(N_CHAINS)),
        ),
    output:
        combined_chains=str(
            MODEL_CACHE_DIR / "{model_name}_STAN_MCMC" / "posterior.netcdf"
        ),
    params:
        n_chains=N_CHAINS,
        combined_cache_dir=str(MODEL_CACHE_DIR),
        config_file=str(MODEL_CONFIG),
        cache_dir=str(TEMP_DIR),
    conda:
        ENVIRONMENT_YAML
    shell:
        "speclet/command_line_interfaces/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  STAN_MCMC"
        "  {params.n_chains}"
        "  {params.config_file}"
        "  {params.cache_dir}"
        "  {params.combined_cache_dir}"


# --- PyMC3 MCMC ---


rule sample_pymc3_mcmc:
    output:
        idata_path=str(
            TEMP_DIR / "{model_name}_PYMC3_MCMC_chain{chain}" / "posterior.netcdf"
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
        get_aesara_flags("{wildcards.model_name}_{wildcards.chain}_mcmc") + " "
        "speclet/command_line_interfaces/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC3_MCMC"
        "  {params.tempdir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --cache-name {params.cache_name}"


rule combine_pymc3_mcmc:
    input:
        chains=expand(
            str(
                TEMP_DIR
                / "{{model_name}}_PYMC3_MCMC_chain{chain}"
                / "posterior.netcdf"
            ),
            chain=list(range(N_CHAINS)),
        ),
    output:
        combined_chains=str(
            MODEL_CACHE_DIR / "{model_name}_PYMC3_MCMC" / "posterior.netcdf"
        ),
    params:
        n_chains=N_CHAINS,
        combined_cache_dir=str(MODEL_CACHE_DIR),
        config_file=str(MODEL_CONFIG),
        cache_dir=str(TEMP_DIR),
    conda:
        ENVIRONMENT_YAML
    shell:
        get_aesara_flags("{wildcards.model_name}_combine-mcmc") + " "
        "speclet/command_line_interfaces/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  PYMC3_MCMC"
        "  {params.n_chains}"
        "  {params.config_file}"
        "  {params.cache_dir}"
        "  {params.combined_cache_dir}"


# --- PyMC3 ADVI ---


rule sample_pymc3_advi:
    output:
        idata_path=str(MODEL_CACHE_DIR / "{model_name}_PYMC3_ADVI" / "posterior.netcdf"),
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
        get_aesara_flags("{wildcards.model_name}_advi") + " "
        "speclet/command_line_interfaces/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC3_ADVI"
        "  {params.cache_dir}"
        "  --mcmc-cores 1"


# --- Summaries ---


rule summarize_posterior:
    input:
        idata_path=MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "posterior.netcdf",
    output:
        description=MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "description.txt",
        posterior_summary=MODEL_CACHE_DIR
        / "{model_name}_{fit_method}"
        / "posterior-summary.csv",
        post_pred=MODEL_CACHE_DIR
        / "{model_name}_{fit_method}"
        / "posterior-predictions.csv",
    params:
        config_file=str(MODEL_CONFIG),
        cache_dir=str(MODEL_CACHE_DIR),
    shell:
        "speclet/command_line_interfaces/summarize_posterior.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  {wildcards.fit_method}"
        "  {params.cache_dir}"
        "  {output.description}"
        "  {output.posterior_summary}"
        "  {output.post_pred}"
        "  --post-pred-thin=40"


rule papermill_report:
    input:
        template_nb=str(REPORTS_DIR / "_model-report-template.ipynb"),
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
        idata_path=str(
            MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "posterior.netcdf"
        ),
        notebook=rules.papermill_report.output.notebook,
    output:
        markdown=str(REPORTS_DIR / "{model_name}_{fit_method}.md"),
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.notebook} && "
        "nbqa isort --profile=black {input.notebook} && "
        "nbqa black {input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"


BENCHMARK_REPORT = "reports/benchmarks.ipynb"
run_benchmark_nb_cmd = f"""
    jupyter nbconvert --to notebook --inplace --execute '{BENCHMARK_REPORT}' &&
    nbqa isort --profile=black '{BENCHMARK_REPORT}' &&
    nbqa black '{BENCHMARK_REPORT}' &&
    jupyter nbconvert --to markdown '{BENCHMARK_REPORT}'
"""


onsuccess:
    shell(run_benchmark_nb_cmd)
