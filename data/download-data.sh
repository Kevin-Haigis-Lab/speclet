#!/bin/bash

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smakemake

SNAKEFILE="data/download-data.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 2 \
    --restart-times 0 \
    --latency-wait 120 \
    --keep-going


conda deactivate
