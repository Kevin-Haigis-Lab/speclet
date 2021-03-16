#!/bin/bash

#SBATCH -c 2
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 5G
#SBATCH -o logs/crc-model-sampling/subsample-ceres-%A.log
#SBATCH -e logs/crc-model-sampling/subsample-ceres-%A.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_smakemake

SNAKEFILE="pipelines/010_010_run-crc-sampling-snakemake.py"

# Copy original env file and ammend import of speclet project modules
ENV_PATH="pipelines/010_014_environment.yml"
cp environment.yml $ENV_PATH
sed -i "s|-e .|-e $(pwd)/|" $ENV_PATH

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 1 \
    --restart-times 0 \
    --latency-wait 120 \
    --use-conda
#     --drmaa " -c {cluster.cores} -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err} -J {cluster.J}"

# --cluster-config pipelines/default_snakemake_config.json \

# to make a dag
# snakemake \
#   --snakefile $SNAKEFILE \
#   --dag |  \
#   dot -Tpdf > analysis/015_snakemake-dag.pdf


conda deactivate
exit 0
