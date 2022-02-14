#!/bin/bash

# Run data downloading pipeline.

module load conda2 slurm-drmaa

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smk

SNAKEFILE="data/download-data.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 2 \
    --restart-times 0 \
    --latency-wait 120 \
    --keep-going

conda deactivate
exit
