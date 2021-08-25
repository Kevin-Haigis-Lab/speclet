#!/bin/bash

#SBATCH --account=park
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 1-00:00
#SBATCH --mem 2G
#SBATCH -o logs/%j-sbc-snakemake.log
#SBATCH -e logs/%j-sbc-snakemake.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate speclet_snakemake

SNAKEFILE="pipelines/012_010_simulation-based-calibration-snakemake.smk"

# Copy original env file and ammend import of speclet project modules
ENV_PATH="pipelines/default_environment.yaml"
if [ ! -f "$ENV_PATH" ]; then
    cp environment.yaml $ENV_PATH
    sed -i "s|-e .|-e $(pwd)/|" $ENV_PATH
    sed -i '/jupyter_contrib_nbextensions/d' $ENV_PATH
fi

snakemake \
    --snakefile $SNAKEFILE \
    --jobs 980 \
    --restart-times 3 \
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
