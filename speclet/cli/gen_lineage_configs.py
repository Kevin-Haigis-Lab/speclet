#!/usr/bin/env python3

"""Auto-generate lineage model configurations."""

import re
from pathlib import Path
from typing import Final

from rich import print as rprint
from typer import Typer

from speclet.io import lineage_modeling_data_dir
from speclet.project_configuration import get_model_configuration_file

app = Typer()


TEMPLATE_CONFIG: Final[
    str
] = """
- name: "hnb-single-lineage-{{LINEAGE}}"
  description: "
    Single lineage hierarchical negative binomial model for {{LINEAGE}} data.
  "
  active: {{ACTIVE}}
  model: LINEAGE_HIERARCHICAL_NB
  data_file: "modeling_data/{{DATA_DIR}}/depmap-modeling-data_{{LINEAGE}}.csv"
  model_kwargs:
    lineage: "{{LINEAGE}}"
    min_n_cancer_genes: {{MIN_N}}
    min_frac_cancer_genes: {{MIN_FRAC}}
  sampling_kwargs:
    pymc_numpyro:
      draws: 1000
      tune: 2000
      target_accept: 0.99
      idata_kwargs:
        log_likelihood: false
      nuts_kwargs:
        step_size: 0.01
  slurm_resources:
    PYMC_NUMPYRO:
      mem: {{MEM}}
      time: {{TIME}}
      cores: 1
      gpu:
        gpu: "RTX 8000"
"""


def _make_config(
    data_dir_name: str,
    lineage: str,
    active: bool,
    memory_gb: int,
    time_hr: int,
    min_n_cancer_genes: int,
    min_frac_cancer_genes: float,
) -> str:
    replacements: Final[dict[str, str]] = {
        "{{DATA_DIR}}": data_dir_name,
        "{{LINEAGE}}": lineage,
        "{{ACTIVE}}": str(active).lower(),
        "{{MIN_N}}": str(min_n_cancer_genes),
        "{{MIN_FRAC}}": str(min_frac_cancer_genes),
        "{{TIME}}": str(time_hr),
        "{{MEM}}": str(memory_gb),
    }
    config = TEMPLATE_CONFIG
    for key, value in replacements.items():
        _config = config.replace(key, value)
        assert _config != config, f"No changes made for '{key}':'{value}'."
        config = _config
    return config


def _list_lineages(data_dir: Path) -> list[str]:
    lineage_files = [f for f in data_dir.iterdir()]
    lineage_files = [f for f in lineage_files if "depmap-modeling-data" in f.name]
    pattern = r"(?<=depmap-modeling-data_).*(?=\.csv)"
    lineages: list[str] = [re.findall(pattern, f.name)[0] for f in lineage_files]
    lineages.sort()
    rprint(f"Found {len(lineages)} lineages.")
    return lineages


def _insert_lineage_configs(config_file: Path, new_text: str) -> None:
    START = "## >>> AUTO GEN LINEAGES >>>\n"
    END = "## <<<\n"
    with open(config_file, "r") as file:
        config_lines = list(file)

    start_i = config_lines.index(START)
    end_i = config_lines.index(END)

    new_config_lines: list[str] = config_lines[: (start_i + 1)]
    new_config_lines += [new_text, "\n"]
    new_config_lines += config_lines[end_i:]
    new_config = "".join(new_config_lines)

    rprint(f"Updating new configuration file '{config_file}'.")
    rprint(f"Changes made between lines {start_i+1}-{end_i+1}.")
    with open(config_file, "w") as file:
        file.write(new_config)
    return None


@app.command()
def generate(
    data_dir: Path | None = None,
    config: Path | None = None,
    active: bool = True,
    memory: int = 24,
    time_hr: int = 8,
    min_n_cancer_genes: int = 3,
    min_frac_cancer_genes: float = 0.05,
) -> None:
    """Autogenerate model configurations for all cell line lineages.

    Inserts model configurations for all lineages with data in the specific folder for
    lineage modeling data files.

    Inserts the configurations between '## >>> AUTO GEN LINEAGES >>>\n' and '## <<<\n'.

    Args:
        data_dir (Path | None, optional): Directory with lineage data files. Defaults
        to `None`.
        config (Path | None, optional): Model configuration file. Defaults to `None`
        which results in using the default configuration in the project configuration.
        active (bool, optional): Mark the new configurations active or inactive.
        Defaults to `True` (active).
        memory (int, optional): Pass the memory (GB) to be used for each configuration.
        Defaults to 32.
        time_hr (int, optional): Pass the time (hours) to be used for each
        configuration. Defaults to 10.
        min_n_cancer_genes (int, optional): Pass the `min_n_cancer_genes` to be used for
        each configuration. Defaults to 4.
        min_frac_cancer_genes (float, optional): Pass the `min_frac_cancer_genes` to be
        used for each configuration. Defaults to 0.05.
    """
    rprint("[blue]Autogenerate cell line lineage configurations.[/blue]")
    if data_dir is None:
        data_dir = lineage_modeling_data_dir()
        rprint(f"[gray]Using lineages from '{data_dir}'.[/gray]")
    if config is None:
        config = get_model_configuration_file()

    lineage_configs = [
        _make_config(
            data_dir_name=data_dir.name,
            lineage=line,
            active=active,
            memory_gb=memory,
            time_hr=time_hr,
            min_n_cancer_genes=min_n_cancer_genes,
            min_frac_cancer_genes=min_frac_cancer_genes,
        )
        for line in _list_lineages(data_dir)
    ]
    _insert_lineage_configs(config, "\n".join(lineage_configs))
    rprint("Done! :party_popper:")
    return None


@app.command()
def clear(config: Path | None = None) -> None:
    """Clear autogenerated model configurations for cell line lineages.

    Clears the text in the configuration file between
    '## >>> AUTO GEN LINEAGES >>>\n' and '## <<<\n'.

    Args:
        config (Path | None, optional): Model configuration file. Defaults to `None`
        which results in using the default configuration in the project configuration.
    """
    rprint("[blue]Clearing cell line lineage configurations.[/blue]")
    if config is None:
        config = get_model_configuration_file()

    _insert_lineage_configs(config, "\n")
    rprint("Done! :ok_hand:")
    return None


if __name__ == "__main__":
    app()
