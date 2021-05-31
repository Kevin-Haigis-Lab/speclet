#!/bin/bash

#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 2G
#SBATCH -o logs/%A_munge_smk.log
#SBATCH -e logs/%A_munge_smk.log

module load gcc conda2 slurm-drmaa/1.1.1 R

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smakemake

./munge/_list_all_depmapids.py

snakemake \
  --snakefile munge/000_prepare-modeling-data.smk \
  --jobs 9950 \
  --restart-times 0 \
  --cluster-config munge/001_prepare-modeling-data_snakemake-config.json \
  --latency-wait 120 \
  --use-conda \
  --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}"

conda deactivate

## Use the following to unlock snakemake after failed runs.
# snakemake \
#   --snakefile munge/010_prepare-modeling-data_snakemake.py \
#   --dry-run \
#   --quiet \
#   --unlock

exit 44
