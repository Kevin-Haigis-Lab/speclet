""" Run a simulation-based calibration for a PyMC3 model."""

from pathlib import Path

import papermill

REPORTS_DIR = Path("reports/crc_sbc_reports/").as_posix()
ENVIRONMENT_YAML = Path("010_014_environment.yml").as_posix()
ROOT_PERMUTATION_DIR = Path("temp/").as_posix()


model_names = ["crc_model_one", "crc_ceres_mimic_one"]


rule all:
    input:
        expand(REPORTS_DIR + "{model_name}_sbc-results.md", model_name=model_names),


rule run_sbc:
    output:
        netcdf_file=(
            ROOT_PERMUTATION_DIR + "{model}/sbc-perm{perm_num}/inference-data.netcdf"
        ),
        posterior_file=(
            ROOT_PERMUTATION_DIR + "{model}/sbc-perm{perm_num}/posterior-summary.csv"
        ),
        priors_file=ROOT_PERMUTATION_DIR + "{model}/sbc-perm{perm_num}/priors.npz",
    shell:
        "pipelines/012_015_run-sbc.py run_sbc "
        "  {wildcards.model} "
        "  " + ROOT_PERMUTATION_DIR + "{model}/sbc-perm{perm_num} "
        "  perm_number={wildcards.perm_num}"


rule papermill_report:
    input:
        template_notebook=REPORTS_DIR + "sbc-results-template.ipynb",
    output:
        notebook=REPORTS_DIR + "{model_name}_sbc-results.ipynb",
    run:
        papermill.execute_notebook(
            input.template_notebook,
            output.notebook,
            parameters={"MODEL": wildcards.model_name},
            prepare_only=True,
        )


rule execute_report:
    input:
        sbc_results,
        notebook=REPORTS_DIR + "{model_name}_sbc-results.ipynb",
    output:
        markdown=REPORTS_DIR + "{model}_{model_name}.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute " + "{input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"


# TODO: how to pass list of permutation files...
# maybe just pass directory of where to look and use `expand()` to list them as an input to the rule.
