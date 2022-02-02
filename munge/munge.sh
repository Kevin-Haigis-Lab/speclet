#!/bin/bash

# Run munge pipeline.

#SBATCH --account=park
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 0-05:00
#SBATCH --mem 2G
#SBATCH -o logs/%j_munge-pipeline.log
#SBATCH -e logs/%j_munge-pipeline.log

module load conda2 slurm-drmaa

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smk

SNAKEFILE="munge/munge.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 9997 \
    --restart-times 0 \
    --keep-going \
    --latency-wait 120 \
    --rerun-incomplete \
    --printshellcmds \
    --use-conda \
    --conda-frontend 'mamba' \
    --drmaa "  --account=park -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}" \
    --cluster-config munge/munge-config.json

conda deactivate
exit 123
