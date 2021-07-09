#!/bin/sh

bash ~/.bashrc

conda create -n speclet_snakemake -f snakemake_environment.yaml

conda create -n speclet -f environment.yaml
conda activate speclet

pre-commit install
