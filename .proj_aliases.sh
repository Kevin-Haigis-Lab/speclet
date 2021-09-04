#!/bin/bash

module unload python
module load R/4.0.1 conda2

# Bash aliases used in this project.
alias speclet_srun="srunp --pty -p priority --mem 80G -c 4 -t 0-15:00 --tunnel 7012:7012 /bin/bash"
alias speclet_env="conda activate speclet && bash .proj_aliases.sh"
alias speclet_jl="jupyter lab --port=7012 --browser='none'"
alias speclet_sshlab="ssh -N -L 7012:127.0.0.1:7012"

alias speclet_snakemake_env="conda activate speclet_snakemake && bash .proj_aliases.sh"
alias smk_fit="snakemake --snakefile pipelines/010_010_run-crc-sampling-snakemake.smk"
alias smk_sbc="snakemake --snakefile pipelines/012_010_simulation-based-calibration-snakemake.smk"
alias smk_date="snakemake --snakefile data/download-data.smk"
alias smk_munge="snakemake --snakefile munge/munge.smk"

# Misc.
alias tmd="jupyter nbconvert --to markdown"
alias nbexec="jupyter nbconvert --to notebook --inplace --execute"
alias pcr="pre-commit run"

# Testing
alias pyt="python3 -m pytest"
alias pyt2="python3 -m pytest --disable-warnings --cov=analysis tests"
