"""Model fitting pipeline."""

from pathlib import Path

import papermill
from snakemake.io import Wildcards

from speclet.managers.slurm_resource_manager import SlurmResourceManager as SlurmRM
from speclet.pipelines.snakemake_parsing_helpers import get_models_names_fit_methods
from speclet.pipelines.aesara_flags import get_aesara_flags
from speclet.project_enums import ModelFitMethod
from speclet.project_configuration import fitting_pipeline_config


pipeline_config = fitting_pipeline_config()

# Global parameters.
N_CHAINS = pipeline_config.num_chains

# Directory and file paths.
TEMP_DIR = pipeline_config.temp_dir
MODEL_CACHE_DIR = pipeline_config.model_cache_dir
REPORTS_DIR = pipeline_config.reports_dir

# Benchmarking jobs.
BENCHMARK_DIR = pipeline_config.benchmark_dir
if not BENCHMARK_DIR.exists():
    BENCHMARK_DIR.mkdir(parents=True)


# --- Model configurations ----

MODEL_CONFIG = pipeline_config.models_config
model_configuration_lists = get_models_names_fit_methods(MODEL_CONFIG)


if len(model_configuration_lists) == 0:
    raise BaseException("No models to run in pipeline in the configuration file.")


# --- Wildcard constrains ----


wildcard_constraints:
    model_name="|".join(set(model_configuration_lists.model_names)),
    fit_method="|".join(a.value for a in ModelFitMethod),
    chain="\d+",


# --- Helpers ----


def create_resource_manager(w: Wildcards, fit_method: ModelFitMethod) -> SlurmRM:
    return SlurmRM(name=w.model_name, fit_method=fit_method, config_path=MODEL_CONFIG)


def get_memory(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).memory


def get_time(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).time


def get_partition(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).partition


def get_gres(w: Wildcards, fit_method: ModelFitMethod) -> str:
    return create_resource_manager(w=w, fit_method=fit_method).gres


# --- Rules ----


localrules:
    all,
    papermill_report,


rule all:
    input:
        report=expand(
            REPORTS_DIR / "{model_name}_{fit_method}.md",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        description=expand(
            MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "description.txt",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        summary=expand(
            MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "posterior-summary.csv",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),
        post_pred=expand(
            MODEL_CACHE_DIR
            / "{model_name}_{fit_method}"
            / "posterior-predictions.csv",
            zip,
            model_name=model_configuration_lists.model_names,
            fit_method=model_configuration_lists.fit_methods,
        ),


# --- PyMC MCMC ---


rule sample_pymc_mcmc:
    output:
        idata_path=TEMP_DIR / "{model_name}_PYMC_MCMC_chain{chain}" / "posterior.netcdf",
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.PYMC_MCMC),
        time=lambda w: get_time(w, ModelFitMethod.PYMC_MCMC),
        partition=lambda w: get_partition(w, ModelFitMethod.PYMC_MCMC),
        config_file=MODEL_CONFIG,
        tempdir=TEMP_DIR,
        cache_name=lambda w: f"{w.model_name}_PYMC_MCMC_chain{w.chain}",
    benchmark:
        BENCHMARK_DIR / "sample_pymc_mcmc/{model_name}_chain{chain}.tsv"
    priority: 20
    shell:
        get_aesara_flags("{wildcards.model_name}_{wildcards.chain}_mcmc") + " "
        "speclet/cli/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC_MCMC"
        "  {params.tempdir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --cache-name {params.cache_name}"
        "  --seed {wildcards.chain}"
        "  --broad-only"


rule combine_pymc_mcmc:
    input:
        chains=expand(
            TEMP_DIR / "{{model_name}}_PYMC_MCMC_chain{chain}" / "posterior.netcdf",
            chain=list(range(N_CHAINS)),
        ),
    output:
        combined_chains=MODEL_CACHE_DIR / "{model_name}_PYMC_MCMC" / "posterior.netcdf",
    params:
        n_chains=N_CHAINS,
        combined_cache_dir=MODEL_CACHE_DIR,
        config_file=MODEL_CONFIG,
        cache_dir=TEMP_DIR,
    shell:
        get_aesara_flags("{wildcards.model_name}_combine-mcmc") + " "
        "speclet/cli/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  PYMC_MCMC"
        "  {params.n_chains}"
        "  {params.config_file}"
        "  {params.cache_dir}"
        "  {params.combined_cache_dir}"


# --- PyMC MCMC Numpyro backend ---


rule sample_pymc_numpyro:
    output:
        idata_path=TEMP_DIR
        / "{model_name}_PYMC_NUMPYRO_chain{chain}"
        / "posterior.netcdf",
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.PYMC_NUMPYRO),
        time=lambda w: get_time(w, ModelFitMethod.PYMC_NUMPYRO),
        partition=lambda w: get_partition(w, ModelFitMethod.PYMC_NUMPYRO),
        gres=lambda w: get_gres(w, ModelFitMethod.PYMC_NUMPYRO),
        config_file=MODEL_CONFIG,
        tempdir=TEMP_DIR,
        cache_name=lambda w: f"{w.model_name}_PYMC_NUMPYRO_chain{w.chain}",
    benchmark:
        BENCHMARK_DIR / "sample_pymc_mcmc/{model_name}_chain{chain}.tsv"
    priority: 30
    retries: 1
    shell:
        get_aesara_flags("{wildcards.model_name}_{wildcards.chain}_mcmc") + " "
        "speclet/cli/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC_NUMPYRO"
        "  {params.tempdir}"
        "  --mcmc-chains 1"
        "  --mcmc-cores 1"
        "  --cache-name {params.cache_name}"
        "  --broad-only"
        "  --log-level DEBUG"
        "  --check-sampling-stats"
        # "  --seed {wildcards.chain}"


rule combine_pymc_numpyro:
    input:
        chains=expand(
            TEMP_DIR / "{{model_name}}_PYMC_NUMPYRO_chain{chain}" / "posterior.netcdf",
            chain=list(range(N_CHAINS)),
        ),
    output:
        combined_chains=MODEL_CACHE_DIR
        / "{model_name}_PYMC_NUMPYRO"
        / "posterior.netcdf",
    params:
        n_chains=N_CHAINS,
        combined_cache_dir=MODEL_CACHE_DIR,
        config_file=MODEL_CONFIG,
        cache_dir=TEMP_DIR,
    shell:
        get_aesara_flags("{wildcards.model_name}_combine-mcmc") + " "
        "speclet/cli/combine_mcmc_chains_cli.py"
        "  {wildcards.model_name}"
        "  PYMC_NUMPYRO"
        "  {params.n_chains}"
        "  {params.config_file}"
        "  {params.cache_dir}"
        "  {params.combined_cache_dir}"


# --- PyMC ADVI ---


rule sample_pymc_advi:
    output:
        idata_path=MODEL_CACHE_DIR / "{model_name}_PYMC_ADVI" / "posterior.netcdf",
    params:
        mem=lambda w: get_memory(w, ModelFitMethod.PYMC_ADVI),
        time=lambda w: get_time(w, ModelFitMethod.PYMC_ADVI),
        partition=lambda w: get_partition(w, ModelFitMethod.PYMC_ADVI),
        config_file=MODEL_CONFIG,
        cache_dir=MODEL_CACHE_DIR,
    benchmark:
        BENCHMARK_DIR / "sample_pymc_advi/{model_name}.tsv"
    priority: 10
    shell:
        get_aesara_flags("{wildcards.model_name}_advi") + " "
        "speclet/cli/fit_bayesian_model_cli.py"
        '  "{wildcards.model_name}"'
        "  {params.config_file}"
        "  PYMC_ADVI"
        "  {params.cache_dir}"
        "  --mcmc-cores 1"
        "  --seed {wildcards.chain}"
        "  --broad-only"


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
        config_file=MODEL_CONFIG,
        cache_dir=MODEL_CACHE_DIR,
    shell:
        "speclet/cli/summarize_posterior.py"
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
        template_nb=REPORTS_DIR / "_model-report-template.ipynb",
    output:
        notebook=REPORTS_DIR / "{model_name}_{fit_method}.ipynb",
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
        idata_path=MODEL_CACHE_DIR / "{model_name}_{fit_method}" / "posterior.netcdf",
        notebook=rules.papermill_report.output.notebook,
    output:
        markdown=REPORTS_DIR / "{model_name}_{fit_method}.md",
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.notebook} && "
        "nbqa isort --profile=black {input.notebook} && "
        "nbqa black {input.notebook} && "
        "jupyter nbconvert --to markdown {input.notebook}"
