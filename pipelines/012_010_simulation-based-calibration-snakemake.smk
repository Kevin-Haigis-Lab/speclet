""" Run a simulation-based calibration for a PyMC3 model."""

from pathlib import Path

import papermill

from pipeline_classes import ModelOption, ModelFitMethod, ModelConfig
from sbc_resource_requirements import SBCResourceManager as RM

NUM_SIMULATIONS = 10

REPORTS_DIR = "reports/crc_sbc_reports/"
ENVIRONMENT_YAML = "default_environment.yml"
ROOT_PERMUTATION_DIR = "/n/scratch3/users/j/jc604/speclet-sbc/"

MOCK_DATA_SIZE = "medium"

model_configurations: List[ModelConfig] = (
    ModelConfig("SpecletSix-mcmc", ModelOption.speclet_six, ModelFitMethod.mcmc)
    ModelConfig("SpecletSix-advi", ModelOption.speclet_six, ModelFitMethod.advi),
    ModelConfig("SpecletSeven-mcmc-noncentered", ModelOption.speclet_seven, ModelFitMethod.mcmc)
    ModelConfig("SpecletSeven-advi-noncentered", ModelOption.speclet_seven, ModelFitMethod.advi),
)

models = [c.model.value for c in model_configurations]
model_names = [c.name for c in model_configurations]
fit_methods = [c.fit_method.value for c in model_configurations]


#### ---- Wildcard constrains ---- ####

wildcard_constraints:
    model="|".join([a.value for a in ModelOption]),
    model_name="|".join(set(model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    perm_num="\d+",


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
        netcdf_file=(
            ROOT_PERMUTATION_DIR
            + "{model}_{model_name}_{fit_method}/sbc-perm{perm_num}/inference-data.netcdf"
        ),
        posterior_file=(
            ROOT_PERMUTATION_DIR
            + "{model}_{model_name}_{fit_method}/sbc-perm{perm_num}/posterior-summary.csv"
        ),
        priors_file=(
            ROOT_PERMUTATION_DIR + "{model}_{model_name}/sbc-perm{perm_num}/priors.npz"
        ),
    conda:
        ENVIRONMENT_YAML
    params:
        cores=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).cores,
        mem=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).memory,
        time=lambda w: RM(w.model, w.model_name, MOCK_DATA_SIZE, w.fit_method).time,
    shell:
        "src/command_line_interfaces/simulation_based_calibration_cli.py "
        "  {wildcards.model} "
        "  {wildcards.model_name} "
        "  {wildcards.fit_method} "
        "  " + ROOT_PERMUTATION_DIR + "{wildcards.model}_{wildcards.model_name}_{fit_method}/sbc-perm{wildcards.perm_num} "
        "  {wildcards.perm_num} "
        " " + MOCK_DATA_SIZE


rule papermill_report:
    output:
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "sbc-results-template.ipynb",
            output.notebook,
            parameters={
                "MODEL": wildcards.model,
                "MODEL_NAME": wildcards.model_name,
                "SBC_RESULTS_DIR": ROOT_PERMUTATION_DIR
                + wildcards.model
                + "_"
                + wildcards.model_name,
                "NUM_SIMULATIONS": NUM_SIMULATIONS,
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        sbc_results=expand(
            ROOT_PERMUTATION_DIR
            + "{model}_{model_name}_{fit_method}/sbc-perm{perm_num}/posterior-summary.csv",
            perm_num=list(range(NUM_SIMULATIONS)),
            allow_missing=True,
        ),
        notebook=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.ipynb",
    output:
        markdown=REPORTS_DIR + "{model}_{model_name}_{fit_method}_sbc-results.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
