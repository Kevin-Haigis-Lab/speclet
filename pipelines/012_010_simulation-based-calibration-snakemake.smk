""" Run a simulation-based calibration for a PyMC3 model."""

from pathlib import Path

import papermill

NUM_SIMULATIONS = 500

REPORTS_DIR = "reports/crc_sbc_reports/"
ENVIRONMENT_YAML = "default_environment.yml"
ROOT_PERMUTATION_DIR = "/n/scratch3/users/j/jc604/speclet-sbc/"


model_names = ["crc_model_one", "crc_ceres_mimic_one"]


rule all:
    input:
        expand(REPORTS_DIR + "{model_name}_sbc-results.md", model_name=model_names),


rule run_sbc:
    output:
        netcdf_file=(
            ROOT_PERMUTATION_DIR + "{model_name}/sbc-perm{perm_num}/inference-data.netcdf"
        ),
        posterior_file=(
            ROOT_PERMUTATION_DIR + "{model_name}/sbc-perm{perm_num}/posterior-summary.csv"
        ),
        priors_file=ROOT_PERMUTATION_DIR + "{model_name}/sbc-perm{perm_num}/priors.npz",
    conda:
        ENVIRONMENT_YAML
    shell:
        "src/command_line_interfaces/simulation_based_calibration_cli.py "
        "  {wildcards.model_name} "
        "  " + ROOT_PERMUTATION_DIR + "{wildcards.model_name}/sbc-perm{wildcards.perm_num} "
        "  {wildcards.perm_num}"
        "  large"


rule papermill_report:
    output:
        notebook=REPORTS_DIR + "{model_name}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "sbc-results-template.ipynb",
            output.notebook,
            parameters={
                "MODEL": wildcards.model_name,
                "SBC_RESULTS_DIR": ROOT_PERMUTATION_DIR + wildcards.model_name,
                "NUM_SIMULATIONS": NUM_SIMULATIONS
            },
            prepare_only=True,
        )


rule execute_report:
    input:
        sbc_results=expand(
            ROOT_PERMUTATION_DIR + "{model_name}/sbc-perm{perm_num}/posterior-summary.csv",
            perm_num=list(range(NUM_SIMULATIONS)),
            allow_missing=True,
        ),
        notebook=REPORTS_DIR + "{model_name}_sbc-results.ipynb",
    output:
        markdown=REPORTS_DIR + "{model_name}_sbc-results.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
