""" Run a simulation-based calibration for a PyMC3 model."""

import os
from pathlib import Path
from typing import List

import papermill
from snakemake.io import Wildcards
import theano

from src.project_enums import ModelFitMethod, ModelOption, SpecletPipeline, MockDataSize
from src.pipelines import snakemake_parsing_helpers as smk_help
from src.pipelines.theano_flags import get_theano_flags
from src.managers.sbc_pipeline_resource_mangement import SBCResourceManager as RM

# SBC parameters.
NUM_SIMULATIONS = 1000
MOCK_DATA_SIZE = MockDataSize.MEDIUM

# Directory and file paths
SCRATCH_DIR = "/n/scratch3/users/j/jc604"
REPORTS_DIR = "reports/crc_sbc_reports/"
ENVIRONMENT_YAML = "default_environment.yaml"
ROOT_PERMUTATION_DIR = f"{SCRATCH_DIR}/speclet-sbc/"
CACHE_DIR = "cache/sbc-cache/"

# Theano compilation locks.
THEANO_FLAG = get_theano_flags(
    unique_id="{wildcards.model_name}_{wildcards.perm_num}",
    theano_compiledir=f"{SCRATCH_DIR}/.theano",
)

# Benchmarking jobs.
BENCHMARK_DIR = Path("benchmarks", "012_010_simulation-based-calibration-snakemake")
if not BENCHMARK_DIR.exists():
    BENCHMARK_DIR.mkdir(parents=True)


#### ---- Model Configurations ---- ####

MODEL_CONFIG = Path("models", "model-configs.yaml")
model_configuration_lists = smk_help.get_models_names_fit_methods(
    MODEL_CONFIG, pipeline=SpecletPipeline.SBC
)

if len(model_configuration_lists) == 0:
    raise BaseException("No models to run in pipeline in the configuration file.")


#### ---- Wildcard constrains ---- ####


wildcard_constraints:
    model_name="|".join(set(model_configuration_lists.model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    perm_num="\d+",


#### ---- Directory management ---- ####

root_perm_dir_template = ROOT_PERMUTATION_DIR + "{model_name}_{fit_method}"
perm_dir_template = "sbc-perm{perm_num}"


def make_root_permutation_directory(w: Wildcards) -> str:
    return root_perm_dir_template.format(
        model_name=w.model_name, fit_method=w.fit_method
    )


def make_permutation_dir(w: Wildcards) -> str:
    return (
        make_root_permutation_directory(w)
        + "/"
        + perm_dir_template.format(perm_num=w.perm_num)
    )


collated_results_template = (
    CACHE_DIR + "{model_name}_{fit_method}_collated-posterior-summaries.pkl"
)


def make_collated_results_path(w: Wildcards) -> str:
    return collated_results_template.format(
        model_name=w.model_name,
        fit_method=w.fit_method,
    )


uniformity_results_template = (
    CACHE_DIR + "{model_name}_{fit_method}_uniformity-test-results.pkl"
)


def make_uniformity_results_path(w: Wildcards) -> str:
    return uniformity_results_template.format(
        model_name=w.model_name,
        fit_method=w.fit_method,
    )


def create_resource_manager(w: Wildcards, attempt: int) -> RM:
    return RM(w.model_name, MOCK_DATA_SIZE, w.fit_method, MODEL_CONFIG, attempt=attempt)


#### ---- Rules ---- ####


localrules:
    all,
    papermill_report,


rule all:
    input:
        expand(
            REPORTS_DIR + "{model_name}_{fit_method}_sbc-results.md",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


rule generate_mockdata:
    version:
        "1"
    output:
        mock_data_path=CACHE_DIR + "{model_name}_mockdata.csv",
    conda:
        ENVIRONMENT_YAML
    params:
        config_path=MODEL_CONFIG.as_posix(),
    benchmark:
        BENCHMARK_DIR / "generate_mockdata/{model_name}.tsv"
    shell:
        "src/command_line_interfaces/simulation_based_calibration_cli.py"
        "  make-mock-data"
        "  {wildcards.model_name}"
        "  {params.config_path}"
        f"  {MOCK_DATA_SIZE.value}"
        "  {output.mock_data_path}"
        "  --random-seed 1234"


rule run_sbc:
    input:
        mock_data_path=rules.generate_mockdata.output.mock_data_path,
    output:
        netcdf_file=(
            root_perm_dir_template + "/" + perm_dir_template + "/inference-data.netcdf"
        ),
        posterior_file=(
            root_perm_dir_template + "/" + perm_dir_template + "/posterior-summary.csv"
        ),
        priors_file=root_perm_dir_template + "/" + perm_dir_template + "/priors.npz",
    conda:
        ENVIRONMENT_YAML
    priority: 20
    params:
        perm_dir=make_permutation_dir,
        config_path=MODEL_CONFIG.as_posix(),
    resources:
        cores=lambda wildcards, attempt: create_resource_manager(
            wildcards, attempt=attempt
        ).cores,
        mem=lambda wildcards, attempt: create_resource_manager(
            wildcards, attempt=attempt
        ).memory,
        time=lambda wildcards, attempt: create_resource_manager(
            wildcards, attempt=attempt
        ).time,
        partition=lambda wildcards, attempt: create_resource_manager(
            wildcards, attempt=attempt
        ).partition,
    benchmark:
        BENCHMARK_DIR / "run_sbc/{model_name}_{fit_method}_perm{perm_num}.tsv"
    shell:
        THEANO_FLAG + " "
        "src/command_line_interfaces/simulation_based_calibration_cli.py"
        "  run-sbc"
        "  {wildcards.model_name}"
        "  {params.config_path}"
        "  {wildcards.fit_method}"
        "  {params.perm_dir}"
        "  {wildcards.perm_num}"
        "  --mock-data-path {input.mock_data_path}"


rule collate_sbc:
    input:
        sbc_results_csvs=expand(
            root_perm_dir_template
            + "/"
            + perm_dir_template
            + "/posterior-summary.csv",
            perm_num=list(range(NUM_SIMULATIONS)),
            allow_missing=True,
        ),
    conda:
        ENVIRONMENT_YAML
    params:
        perm_dir=make_root_permutation_directory,
    output:
        collated_results=collated_results_template,
    priority: 10
    benchmark:
        BENCHMARK_DIR / "collate_sbc/{model_name}_{fit_method}.tsv"
    shell:
        "src/command_line_interfaces/collate_sbc_cli.py "
        " collate-sbc-posteriors-cli"
        " {params.perm_dir} "
        " {output.collated_results} "
        " --num-permutations=" + str(NUM_SIMULATIONS)


rule sbc_uniformity_test:
    input:
        sbc_results_csvs=expand(
            root_perm_dir_template
            + "/"
            + perm_dir_template
            + "/posterior-summary.csv",
            perm_num=list(range(NUM_SIMULATIONS)),
            allow_missing=True,
        ),
    conda:
        ENVIRONMENT_YAML
    params:
        perm_dir=make_root_permutation_directory,
    output:
        uniformity_results=uniformity_results_template,
    priority: 10
    benchmark:
        BENCHMARK_DIR / "sbc_uniformity_test/{model_name}_{fit_method}.tsv"
    shell:
        "src/command_line_interfaces/collate_sbc_cli.py "
        " uniformity-test-results"
        " {params.perm_dir} "
        " {output.uniformity_results} "
        " --num-permutations=" + str(NUM_SIMULATIONS)


rule papermill_report:
    input:
        REPORTS_DIR + "sbc-results-template.ipynb",
    version:
        "3"
    params:
        root_perm_dir=make_root_permutation_directory,
        collated_results=make_collated_results_path,
        uniformity_results=make_uniformity_results_path,
    output:
        notebook=REPORTS_DIR + "{model_name}_{fit_method}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "sbc-results-template.ipynb",
            output.notebook,
            parameters={
                "MODEL_NAME": wildcards.model_name,
                "SBC_RESULTS_DIR": params.root_perm_dir,
                "SBC_COLLATED_RESULTS": params.collated_results,
                "SBC_UNIFORMITY_RESULTS": params.uniformity_results,
                "NUM_SIMULATIONS": NUM_SIMULATIONS,
                "CONFIG_PATH": MODEL_CONFIG.as_posix(),
                "FIT_METHOD_STR": wildcards.fit_method,
            },
            prepare_only=True,
        )


rule execute_report:
    version:
        "2"
    input:
        collated_results=rules.collate_sbc.output.collated_results,
        uniformity_results=rules.sbc_uniformity_test.output.uniformity_results,
        notebook=rules.papermill_report.output.notebook,
    output:
        markdown=REPORTS_DIR + "{model_name}_{fit_method}_sbc-results.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
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
