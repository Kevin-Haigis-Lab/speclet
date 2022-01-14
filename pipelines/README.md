# Pipelines

To improve the efficiency and reproducibility of this project's analyses, the long running computations are primarily performed in pipelines.
Thus, the most computationally-intensive parts of the project are parallelizable and scalable.
The ['Snakemake'](https://snakemake.readthedocs.io/en/stable/index.html) workflow management system ensures that only the required computations are performed when a pipeline is run and is able to monitor the different jobs over a HPC (such as O2).

Quick, diagnostic results of the pipelines are saved to the ["reports/"](../reports) directory.

## Setup

The virtual environment must be created before running any of the pipelines.

```bash
conda install -n base -c conda-forge mamba
mamba env create -f environment_smk.yaml
conda activate speclet_smk
# Alternatively
make pyenvs
```

## Pipelines

### 010. Modeling fitting pipeline

This pipeline fits the Bayesian models according to the specifications in ["models/model-configs.yaml"](models/model-configs.yaml).
The results are stored in the same directory for later analysis.

Below are the descriptions of the relevant files:

1. `010_010_model-fitting-pipeline.smk`: Snakemake pipeline
2. `010_011_smk-config.json`: SLURM configuration
3. `010_012_run-model-fitting-pipeline.sh`: Bash script to run the pipeline

The pipeline can be run using the following make command.

```bash
make fit
```

### 012. Simulation-based calibration (SBC) pipeline

> **[WIP] This pipeline needs to be update for the new structure of 'speclet.'**

This pipeline performs simulation-based calibration (SBC) for models according to the specifications in ["models/model-configs.yaml"](models/model-configs.yaml).
Briefly, SBC is used to demonstrate that a model performs as expected by creating mock data from distributions defined by parameters sampled from the model's prior distribution and then fitting the model, expecting to recover the known parameters.

Below are the descriptions of the relevant files:

1. `012_010_simulation-based-calibration.smk`: Snakemake pipeline
2. `012_011_smk-config.json`: SLURM configuration
3. `012_012_run-simulation-based-calibration.sh`: Bash script to run the pipeline

The pipeline can be run using the following make command.

```bash
make sbc
```
