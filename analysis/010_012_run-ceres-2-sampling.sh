#!/bin/bash

#SBATCH -c 3
#SBATCH -p priority
#SBATCH -t 4-00:00
#SBATCH --mem 200G
#SBATCH -o logs/ceres-models/subsample-ceres-%A.log
#SBATCH -e logs/ceres-models/subsample-ceres-%A.log

module unload python
module load gcc conda2

bash ~/.bashrc
conda activate speclet

#python3 analysis/sampling_pymc3_models.py -m "ceres-m1" --force-sample -d
python3 analysis/sampling_pymc3_models.py -m "ceres-m2" --force-sample -d

conda deactivate

exit 0
