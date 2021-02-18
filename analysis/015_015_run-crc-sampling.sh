#!/bin/bash

#SBATCH -c 3
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 200G
#SBATCH -o logs/crc-model-sampling/subsample-ceres-%A.log
#SBATCH -e logs/crc-model-sampling/subsample-ceres-%A.log

module unload python
module load gcc conda2

source /home/jc604/.bashrc
conda activate speclet

python3 analysis/sampling_pymc3_models.py \
    --model "crc-m1" \
    --name "CRC_test_model1" \
    --force-sample \
    --debug

conda deactivate
exit 0
