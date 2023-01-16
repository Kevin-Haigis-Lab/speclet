# Pipelines

To improve the efficiency and reproducibility of this project's analyses, the long running computations are performed in pipelines.
Thus, the most computationally-intensive parts of the project are parallelizable and scalable.
The ['Snakemake'](https://snakemake.readthedocs.io/en/stable/index.html) workflow management system ensures that only the required computations are performed when a pipeline is run and is able to monitor the different jobs over a HPC (such as O2).

The main pipeline in this project fits the models for each cell line lineage.
Overview diagnostic results of the model fitting pipeline are saved to the ["reports/"](../reports) directory.

## Setup

See the primary README for how to setup the development environment.

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

---

## Miscellaneous

On O2, I linked the "temp/" directory to Scratch.

```bash
ln -s $JHC_SCRATCH/speclet-temp temp
```
