"""Command line interface for merging MCMC chains for a single model."""

from pathlib import Path
from typing import List, Optional

import arviz as az
import typer

from src.command_line_interfaces import cli_helpers
from src.command_line_interfaces.cli_helpers import ModelOption
from src.io import cache_io
from src.loggers import logger

app = typer.Typer()
cli_helpers.configure_pretty()


def _get_model_names_for_separate_chains(name: str, n_chains: int) -> List[str]:
    return [f"{name}_chain{i}" for i in range(n_chains)]


@app.command()
def combine_mcmc_chains(
    model: ModelOption,
    name: str,
    chains: List[Path],
    touch_file: Optional[Path] = None,
    debug: bool = False,
) -> None:
    """Combine multiple MCMC chains for a single model.

    Args:
        model (ModelOption): SpecletModel that the chains correspond to.
        name (str): Name of the model.
        chains (List[Path]): The paths to the individual chains (not used).
        debug (bool, optional): Is the model in debug mode? Defaults to False.

    Raises:
        TypeError: Raised if something has gone wrong in the merge.
    """
    if debug:
        logger.info("Setting up model in debug mode.")

    name = cli_helpers.clean_model_names(name)

    n_chains = len(chains)
    logger.info(f"Expecting {n_chains} separate chains.")

    ModelClass = cli_helpers.get_model_class(model_opt=model)
    logger.info(f"Detected model: '{ModelClass.__name__}'")

    model_names = _get_model_names_for_separate_chains(name, n_chains)
    logger.debug(f"Expecting the following model names: {model_names}")

    all_arviz_data_objects: List[az.InferenceData] = []
    for chain_name in model_names:
        logger.info(f"Gather MCMC chain '{chain_name}'.")
        sp_model = ModelClass(
            name=chain_name, root_cache_dir=cache_io.default_cache_dir(), debug=debug
        )

        logger.debug(f"Cache directory: {sp_model.cache_manager.cache_dir.as_posix()}")
        logger.debug(
            f"Cache directory exists: {sp_model.cache_manager.cache_dir.exists()}"
        )
        _mcmc_cache_dir = sp_model.cache_manager.mcmc_cache_delegate.cache_dir
        logger.debug(f"MCMC cache directory: {_mcmc_cache_dir.as_posix()}")
        logger.debug(f"MCMC cache directory exists: {_mcmc_cache_dir.exists()}")

        cli_helpers.modify_model_by_name(model=sp_model, name=name)
        if sp_model.cache_manager.mcmc_cache_exists():
            logger.info(f"Found chain '{chain_name}'.")
            sp_model.build_model()
            assert sp_model.model is not None
            mcmc_res = sp_model.mcmc_sample_model()
            all_arviz_data_objects.append(mcmc_res)
        else:
            logger.error(
                f"Cache for SpecletModel '{sp_model.name}' does not exist - skipping."
            )

    if len(all_arviz_data_objects) != n_chains:
        logger.error(f"Expected {n_chains} but found {len(all_arviz_data_objects)}.")
    else:
        logger.info(f"Collected {len(all_arviz_data_objects)} chains.")

    combined_chains = az.concat(all_arviz_data_objects, dim="chain")
    combined_sp_model = ModelClass(
        name=name, root_cache_dir=cache_io.default_cache_dir(), debug=debug
    )
    if isinstance(combined_chains, az.InferenceData):
        combined_sp_model.mcmc_results = combined_chains
    else:
        logger.error(f"`combined_chains` of type '{type(combined_chains)}'")
        raise TypeError("`combined_chains` was not of type 'arviz.InferenceData'.")
    combined_sp_model.write_mcmc_cache()

    logger.info(f"Finished combining chains for model '{combined_sp_model.name}'")

    if touch_file is not None:
        logger.info(f"Touching file: '{touch_file.as_posix()}'.")
        touch_file.touch()

    return None


if __name__ == "__main__":
    app()
