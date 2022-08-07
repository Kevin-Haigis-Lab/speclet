#!/usr/bin/env python3

"""Summarize a model's posterior sample."""

import json
from inspect import getdoc
from pathlib import Path

import arviz as az
import pandas as pd
from dotenv import load_dotenv
from typer import Typer

from speclet import model_configuration as model_config
from speclet.analysis.arviz_analysis import describe_mcmc
from speclet.bayesian_models import BayesianModelProtocol, get_bayesian_model
from speclet.cli import cli_helpers
from speclet.loggers import logger
from speclet.managers.cache_manager import (
    get_cached_posterior,
    get_posterior_cache_name,
)
from speclet.model_configuration import BayesianModelConfiguration
from speclet.project_configuration import get_bayesian_modeling_constants
from speclet.project_enums import ModelFitMethod

# ---- Setup ----

load_dotenv()
cli_helpers.configure_pretty()
app = Typer()


def _hdi_prob() -> float:
    return get_bayesian_modeling_constants().hdi_prob


def _posterior_description(
    diagnostic_path: Path,
    model: BayesianModelProtocol,
    name: str,
    config: BayesianModelConfiguration,
    trace: az.InferenceData,
    fit_method: ModelFitMethod,
) -> None:
    output = ""
    br = "\n\n" + "-" * 80 + "\n\n"

    model_doc = getdoc(model)
    if model_doc is None:
        model_doc = "(None)"

    output += f"config. name: '{name}'\n"
    output += f"model name: '{type(model).__name__}'\n"
    output += f"model version: '{model.version}'\n"
    output += f"model description: {model_doc}"
    output += f"fit method: '{fit_method.value}'"

    output += br

    logger.info("Copying configuration.")
    output += "CONFIGURATION\n\n"
    output += json.dumps(json.loads(config.json()), indent=4)

    output += br

    if (posterior := trace.get("posterior")) is not None:
        logger.info("Recording posterior.")
        output += "POSTERIOR\n\n"
        output += str(posterior)
    else:
        logger.warning("No posterior found.")
        output += "(No posterior found.)"

    output += br

    if (sample_stats := trace.get("sample_stats")) is not None:
        logger.info("Recording sample stats.")
        output += "SAMPLE STATS\n\n"
        output += str(sample_stats)
    else:
        logger.warning("No sampling stats found.")
        output += "(No sampling stats found.)"

    output += br

    if fit_method in {ModelFitMethod.PYMC_MCMC, ModelFitMethod.PYMC_NUMPYRO}:
        logger.info("Recording MCMC description.")
        output += "MCMC DESCRIPTION\n\n"
        mcmc_desc = describe_mcmc(trace, silent=True, plot=False)
        output += str(mcmc_desc)

    logger.info(f"Writing posterior description to '{str(diagnostic_path)}'.")
    with open(diagnostic_path, "w") as file:
        file.write(output)

    logger.info("Finished description output.")
    return None


def _posterior_summary(
    posterior_summary_path: Path,
    trace: az.InferenceData,
    vars_regex: list[str] | None = None,
) -> None:
    logger.info("Summarizing model posterior.")
    post_summ = az.summary(
        trace, var_names=vars_regex, filter_vars="regex", hdi_prob=_hdi_prob()
    )
    assert isinstance(post_summ, pd.DataFrame)
    logger.info(f"Writing posterior summary to '{str(posterior_summary_path)}'.")
    post_summ.to_csv(posterior_summary_path, index_label="parameter")
    logger.info("Finished writing posterior summary.")
    return None


def _posterior_predictions(
    post_pred_summary_path: Path,
    trace: az.InferenceData,
    thin: int,
) -> None:
    logger.info("Retrieving posterior predictive distribution.")
    ppc = trace.get("posterior_predictive")
    if ppc is None:
        logger.error("Model posterior predictions not found.")
        return None

    # Only expecting a single post. pred. variable. Can change to a more flexible system
    # if needed.
    n_post_preds = len(ppc.keys())
    if n_post_preds != 1:
        raise BaseException(f"Only 1 post. pred. expected; found {n_post_preds}")

    logger.info(f"Writing posterior predictions to '{str(post_pred_summary_path)}'.")
    # Iterating through a collection of length 1.
    for ppc_ary in ppc.values():
        ppc_ary[:, ::thin, :].to_dataframe().to_csv(post_pred_summary_path)

    logger.info("Finished writing posterior predictions.")
    return None


@app.command()
def summarize_posterior(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    description_path: Path,
    posterior_summary_path: Path,
    post_pred_path: Path,
    post_pred_thin: int = 50,
    cache_name: str | None = None,
) -> None:
    """Summarize a model posterior.

    Args:
        name (str): Name of the model (corresponding to a model configuration).
        config_path (Path): Path the the model configuration file.
        fit_method (ModelFitMethod): Method used to fit the model.
        cache_dir (Path): Directory where the posterior file is cached.
        description_path (Path): Path to write the description file.
        posterior_summary_path (Path): Path to write the posterior summary CSV.
        post_pred_path (Path): Path to write the posterior predictions as a CSV.
        post_pred_thin (int, optional): Step-size for thinning posterior predictive
        draws. Defaults to 50. Will end up with the number of draws divided by the step
        size.
        cache_name (Optional[str], optional): Cache name to override the default one
        built with the model and name fit method. Defaults to None.
    """
    logger.info("Reading model configuration.")
    config = model_config.get_configuration_for_model(
        config_path=config_path, name=name
    )
    assert config is not None

    if cache_name is None:
        logger.warning("No cache name provided - one will be generated automatically.")
        cache_name = get_posterior_cache_name(model_name=name, fit_method=fit_method)

    logger.info("Retrieving Bayesian model object.")
    model = get_bayesian_model(config.model)(**config.model_kwargs)

    logger.info("Reading model posterior from file.")
    trace = get_cached_posterior(id=cache_name, cache_dir=cache_dir)

    _posterior_description(
        description_path,
        model=model,
        name=name,
        config=config,
        trace=trace,
        fit_method=fit_method,
    )
    _posterior_summary(
        posterior_summary_path, trace=trace, vars_regex=model.vars_regex(fit_method)
    )
    _posterior_predictions(post_pred_path, trace=trace, thin=post_pred_thin)
    return None


if __name__ == "__main__":
    app()
