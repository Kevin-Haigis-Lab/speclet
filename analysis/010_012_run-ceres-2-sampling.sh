#!/bin/bash

#SBATCH -c 7
#SBATCH -p priority
#SBATCH -t 0-05:00
#SBATCH --mem 50G
#SBATCH -o logs/ceres-models/subsample-ceres-%A.log
#SBATCH -e logs/ceres-models/subsample-ceres-%A.log

module unload python
module load gcc conda2

conda activate speclet

srun -c 3 python3 analysis/010_012_ceres-2-sampling.py -m "ceres-m1" --force-sample -d &
srun -c 3 python3 analysis/010_012_ceres-2-sampling.py -m "ceres-m2" --force-sample -d &
wait

conda deactivate

exit 0
