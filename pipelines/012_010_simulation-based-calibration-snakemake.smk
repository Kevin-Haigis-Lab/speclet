""" Run a simulation-based calibration for a PyMC3 model."""

from pathlib import Path

import papermill

N_PERMUTATIONS = 5

REPORTS_DIR = "reports/crc_sbc_reports/"
ENVIRONMENT_YAML = "default_environment.yml"
ROOT_PERMUTATION_DIR = "temp/"


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
        "pipelines/012_015_run-sbc.py "
        "  {wildcards.model_name} "
        "  " + ROOT_PERMUTATION_DIR + "{wildcards.model_name}/sbc-perm{wildcards.perm_num} "
        "  {wildcards.perm_num}"


rule papermill_report:
    output:
        notebook=REPORTS_DIR + "{model_name}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            REPORTS_DIR + "sbc-results-template.ipynb",
            output.notebook,
            parameters={"MODEL": wildcards.model_name},
            prepare_only=True,
        )


rule execute_report:
    input:
        sbc_results=expand(
            ROOT_PERMUTATION_DIR + "{model_name}/sbc-perm{perm_num}/posterior-summary.csv",
            perm_num=list(range(N_PERMUTATIONS)),
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


# TODO: how to pass list of permutation files...
# maybe just pass directory of where to look and use `expand()` to list them as an input to the rule.
