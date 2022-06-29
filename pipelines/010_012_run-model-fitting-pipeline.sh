#!/bin/bash

# Run model fitting pipeline.

#SBATCH --job-name=fit-pipe
#SBATCH --account=park
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 4G
#SBATCH -o logs/%j_sample-pipeline.log
#SBATCH -e logs/%j_sample-pipeline.log

module load gcc/6.2.0 slurm-drmaa/1.1.3 conda2

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smk

SNAKEFILE="pipelines/010_010_model-fitting-pipeline.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 9995 \
    --restart-times 0 \
    --latency-wait 120 \
    --rerun-incomplete \
    --drmaa " --account=park -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}" \
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
