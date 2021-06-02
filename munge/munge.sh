#!/bin/bash

#SBATCH -c 2
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 8G
#SBATCH -o logs/%j_munge-pipeline.log
#SBATCH -e logs/%j_munge-pipeline.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_snakemake

# Make a list of all DepMap IDs for Snakemake.
./munge/_list_all_depmapids.R

SNAKEFILE="munge/munge.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 2 \
    --restart-times 0 \
    --latency-wait 120 \
    --keep-going

# snakemake \
#     --snakefile $SNAKEFILE \
#     --jobs 20 \
#     --restart-times 0 \
#     --latency-wait 120 \
#     --use-conda \
#     --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}" \
#     --cluster-config pipelines/010_011_smk-config.json \
#     --keep-going


conda deactivate
