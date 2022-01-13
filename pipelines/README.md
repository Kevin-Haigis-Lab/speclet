# Pipelines

To improve the efficiency and reproducibility of this project's analyses, the long running computations are primarily performed in pipelines.
Thus, the most computationally-intensive parts of the project are parallelizable and scalable.
The 'Snakemake' workflow management system ensures that only the required computations are performed when a pipeline is run and is able to monitor the different jobs over a HPC (such as O2).

The pipelines interact with the  modules in `src`, `src.command_line_interfaces` and `src.pipelines`.
Some care had to be taken to ensure that the modules imported by the Snakemake pipelines depend on libraries in the virtual environment used for the pipelines (`speclet_snakemake`).

The results of the pipelines are saved to the ["reports/"](../reports) directory.

## Setup

The virtual environment must be created before running any of the pipelines.

```bash
conda install -n base -c conda-forge mamba
mamba env create -f environment_smk.yaml
conda activate speclet_smk
```

## Pipelines

### 010. Modeling fitting pipeline

This pipeline fits the models included in the speclet `src.models` submodule.

Below are the descriptions of the relevant files:

1. `010_010_run-crc-sampling-snakemake.smk` – Snakemake pipeline.
2. `010_011_smk-config.json` – SLURM configuration.
3. `010_012_run-crc-sampling.sh` – Bash script to run the pipeline.

The pipeline can be run using the following make command.

```bash
make fit
```

### 012. Simulation-based calibration (SBC) pipeline

This pipeline performs simulation-based calibration (SBC) for models included in the speclet `src.models` submodule.
Briefly, SBC is used to demonstrate that a model performs as expected by creating mock data from distributions defined by parameters sampled from the model's prior distribution and then fitting the model, expecting to recover the known parameters.

Below are the descriptions of the relevant files:

1. `012_010_simulation-based-calibration-snakemake.smk` – Snakemake pipeline.
2. `012_011_smk-config.json` – SLURM configuration.
3. `012_012_simulation-based-calibration.sh` – Bash script to run the pipeline.

The pipeline can be run using the following make command.

```bash
make sbc
```
