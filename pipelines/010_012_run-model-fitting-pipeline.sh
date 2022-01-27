#!/bin/bash

#SBATCH --account=park
#SBATCH -c 2
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 4G
#SBATCH -o logs/%j_sample-pipeline.log
#SBATCH -e logs/%j_sample-pipeline.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smk

SNAKEFILE="pipelines/010_010_model-fitting-pipeline.smk"

ENV_PATH="pipelines/default_environment.yaml"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 9995 \
    --restart-times 0 \
    --latency-wait 120 \
    --rerun-incomplete \
    --use-conda \
    --conda-frontend 'mamba' \
    --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}" \
    --cluster-config pipelines/010_011_smk-config.yaml \
    --keep-going \
    --printshellcmds

# --conda-cleanup-envs  # use to clean up old conda envs

# to make a dag
# snakemake \
#   --snakefile $SNAKEFILE \
#   --dag |  \
#   dot -Tpdf > analysis/015_snakemake-dag.pdf

conda deactivate
exit 4
