#!/bin/bash

#SBATCH -c 3
#SBATCH -p priority
#SBATCH -t 0-00:30
#SBATCH --mem 16G
#SBATCH -o logs/ceres-models/subsample-test-%A.log
#SBATCH -e logs/ceres-models/subsample-test-%A.log

module unload python
module load gcc conda2

bash ~/.bashrc
conda activate speclet

jupyter nbconvert --to notebook --inplace --execute "$1"
jupyter nbconvert --to markdown "$1"

conda deactivate

exit 0
