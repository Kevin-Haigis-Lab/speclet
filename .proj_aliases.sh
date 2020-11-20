#!/bin/bash

module unload python
module load R/4.0.1 conda2

# Bash aliases used in this project.
alias speclet_srun="srun --pty -p priority --mem 80G -c 6 -t 0-15:00 /bin/bash"
alias speclet_env="conda activate speclet && bash .proj_aliases.sh"
alias speclet_jl="jupyter lab --port=7012 --browser='none'"
alias speclet_sshlab='ssh -N -L 7012:127.0.0.1:7012'

alias speclet_snakemake_env="conda activate speclet_smakemake && bash .proj_aliases.sh"

# Misc.
alias tmd='jupyter nbconvert --to markdown'
alias pcr='pre-commit run'
