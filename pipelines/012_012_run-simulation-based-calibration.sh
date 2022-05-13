#!/bin/bash

# Run SBC pipeline.

#SBATCH --account=park
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 2G
#SBATCH -o logs/%j-sbc-snakemake.log
#SBATCH -e logs/%j-sbc-snakemake.log

module load conda2 slurm-drmaa

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smk

SNAKEFILE="pipelines/012_010_simulation-based-calibration.smk"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 9990 \
    --restart-times 1 \
    --latency-wait 120 \
    --use-conda \
    --keep-going \
    --printshellcmds \
    --cluster-config pipelines/012_011_smk-config.yaml \
    --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}"

# --conda-cleanup-envs  # use to clean up old conda envs

# to make a dag
# snakemake \
#   --snakefile $SNAKEFILE \
#   --dag |  \
#   dot -Tpdf > analysis/015_snakemake-dag.pdf

conda deactivate
exit 44
