#!/bin/bash

#SBATCH -c 3
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 70G
#SBATCH -o logs/ceres-models/subsample-ceres-%A.log
#SBATCH -e logs/ceres-models/subsample-ceres-%A.log

module unload python
module load gcc conda2

source /home/jc604/.bashrc
conda activate speclet

#python3 analysis/010_012_ceres-2-sampling.py -m "ceres-m1" --force-sample -d
python3 analysis/010_012_ceres-2-sampling.py -m "ceres-m2" --force-sample -d

conda deactivate

exit 0
