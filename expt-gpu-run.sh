#!/bin/bash

#SBATCH --job-name=expt-gpu
#SBATCH --account=park
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem 64G
#SBATCH -o logs/%j_expt-gpu.log
#SBATCH -e logs/%j_expt-gpu.log

module load gcc/9.2.0 cuda/11.2 conda2

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate

# The "job_gpu_monitor.sh" script is provided by RC to help monitor GPU usage.
/n/cluster/bin/job_gpu_monitor.sh & ./speclet/cli/fit_bayesian_model_cli.py \
    "hnb-single-lineage-prostate" \
    models/model-configs.yaml \
    "PYMC_NUMPYRO" \
    temp/ \
    --mcmc-chains 1 \
    --mcmc-cores 2 \
    --seed 123 \
    --broad-only

exit 23
