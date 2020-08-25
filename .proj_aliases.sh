#!/bin/bash

module load R/4.0.1 python/3.7.4 conda2

# Bash aliases used in this project.
alias speclet_srun="srun --pty -p priority --mem 50G -c 5 -t 0-18:00 --x11 /bin/bash"
alias speclet_env="conda activate speclet && bash .proj_aliases.sh"
alias speclet_jl="jupyter lab --port=7012 --browser='none'"
alias speclet_sshlab='ssh -N -L 7012:127.0.0.1:7012'
