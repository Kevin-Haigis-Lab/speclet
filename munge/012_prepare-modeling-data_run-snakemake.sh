#!/bin/sh

#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 10G
#SBATCH -o logs/modeling-data-prep/snakemake_%A.log
#SBATCH -e logs/modeling-data-prep/snakemake_%A.log

module load gcc conda2 slurm-drmaa/1.1.1

bash ~/.bashrc
conda activate speclet_smakemake

snakemake \
  --snakefile munge/010_prepare-modeling-data_snakemake.py \
  --jobs 9950 \
  --restart-times 0 \
  --cluster-config munge/011_prepare-modeling-data_snakemake-config.json \
  --latency-wait 120 \
  --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}"

conda deactivate

## Use the following to unlock snakemake after failed runs.
# snakemake \
#   --snakefile munge/010_prepare-modeling-data_snakemake.py \
#   --dry-run \
#   --quiet \
#   --unlock
