#!/bin/bash

#SBATCH --account=park
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 2G
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
    --jobs 9997 \
    --restart-times 0 \
    --latency-wait 120 \
    --printshellcmds \
    --use-conda \
    --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}" \
    --cluster-config munge/munge-config.json \
    --keep-going

conda deactivate
exit 44
