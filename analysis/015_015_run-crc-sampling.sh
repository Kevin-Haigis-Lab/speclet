#!/bin/bash

#SBATCH -c 3
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 200G
#SBATCH -o logs/crc-model-sampling/subsample-ceres-%A.log
#SBATCH -e logs/crc-model-sampling/subsample-ceres-%A.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smakemake

SNAKEFILE="analysis/015_017_run-crc-sampling-snakemake.py"

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 1 \
    --restart-times 0 \
    --latency-wait 120 \
    --use-conda
#     --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}"

# --cluster-config config/default_snakemake_config.json \

# to make a dag
# snakemake \
#   --snakefile $SNAKEFILE \
#   --dag |  \
#   dot -Tpdf > analysis/015_snakemake-dag.pdf


conda deactivate
exit 0
