""" Run a simulation-based calibration for a PyMC3 model."""

from pathlib import Path
from typing import List

import papermill

from src.project_enums import ModelFitMethod
from src.pipelines.pipeline_classes import ModelOption, ModelConfig
from src.managers.sbc_pipeline_resource_mangement import SBCResourceManager as RM

NUM_SIMULATIONS = 3

REPORTS_DIR = "reports/crc_sbc_reports/"
ENVIRONMENT_YAML = "default_environment.yml"
ROOT_PERMUTATION_DIR = "/n/scratch3/users/j/jc604/speclet-sbc/"

MOCK_DATA_SIZE = "small"

model_configurations: List[ModelConfig] = [
    ModelConfig(name="SpecletSix-mcmc", model=ModelOption.speclet_six, fit_method=ModelFitMethod.mcmc),
    ModelConfig(name="SpecletSix-advi", model=ModelOption.speclet_six, fit_method=ModelFitMethod.advi),
    ModelConfig(name="SpecletSeven-mcmc-noncentered", model=ModelOption.speclet_seven, fit_method=ModelFitMethod.mcmc),
    ModelConfig(name="SpecletSeven-advi-noncentered", model=ModelOption.speclet_seven, fit_method=ModelFitMethod.advi),
]

models = [c.model.value for c in model_configurations]
model_names = [c.name for c in model_configurations]
fit_methods = [c.fit_method.value for c in model_configurations]


#### ---- Wildcard constrains ---- ####

wildcard_constraints:
    model="|".join([a.value for a in ModelOption]),
    model_name="|".join(set(model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    perm_num="\d+",


#### ---- Directory management ---- ####

root_perm_dir_template = ROOT_PERMUTATION_DIR + "{model}_{model_name}_{fit_method}"
perm_dir_template = "sbc-perm{perm_num}"

def make_root_permutation_directory(w) -> str:
    return root_perm_dir_template.format(
        model=w.model,
        model_name=w.model_name,
        fit_method=w.fit_method
    )

def make_permutation_dir(w) -> str:
    return make_root_permutation_directory(w) + "/" + perm_dir_template.format(perm_num=w.perm_num)

collated_results_template = "cache/sbc-cache/{model}_{model_name}_{fit_method}_collated-posterior-summaries.pkl"

def make_collated_results_path(w) -> str:
    print(type(w))
    return collated_results_template.format(
        model=w.model,
        model_name=w.model_name,
        fit_method=w.fit_method,
    )

#### ---- Rules ---- ####

rule all:
    input:
        expand(
            REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.md",
            zip,
            model=models,
            model_name=model_names,
            fit_method=fit_methods,
        ),


rule run_sbc:
    output:
        netcdf_file=root_perm_dir_template + "/" + perm_dir_template + "/inference-data.netcdf",
        posterior_file=root_perm_dir_template + "/" + perm_dir_template + "/posterior-summary.csv",
        priors_file=root_perm_dir_template + "/" + perm_dir_template + "/priors.npz",
    conda:
        ENVIRONMENT_YAML
    params:
        cores=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).cores,
        mem=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).memory,
        time=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).time,
        perm_dir=make_root_permutation_directory,
    shell:
        "src/command_line_interfaces/simulation_based_calibration_cli.py "
        "  {wildcards.model} "
        "  {wildcards.model_name} "
        "  {wildcards.fit_method} "
        "  {params.perm_dir} "
        "  {wildcards.perm_num} "
        " " + MOCK_DATA_SIZE

rule collate_sbc:
    input:
        sbc_results_csvs=expand(
            root_perm_dir_template + "/" + perm_dir_template + "/posterior-summary.csv",
            perm_num=list(range(NUM_SIMULATIONS)),
            allow_missing=True,
        ),
    conda:
        ENVIRONMENT_YAML
    params:
        perm_dir=make_root_permutation_directory,
    output:
        collated_results=collated_results_template,
    shell:
        "src/command_line_interfaces/collate_sbc_cli.py "
        " {params.perm_dir} "
        " {output.collated_results} "
        " --num-permutations=" + str(NUM_SIMULATIONS)

rule papermill_report:
    params:
        root_perm_dir=make_root_permutation_directory,
        collated_results=make_collated_results_path,
    output:
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "sbc-results-template.ipynb",
            output.notebook,
            parameters={
                "MODEL": wildcards.model,
                "MODEL_NAME": wildcards.model_name,
                "SBC_RESULTS_DIR": params.root_perm_dir,
                "SBC_COLLATED_RESULTS": params.collated_results,
                "NUM_SIMULATIONS": NUM_SIMULATIONS,
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        collated_results=collated_results_template,
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.ipynb",
    output:
        markdown=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
