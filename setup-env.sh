#!/bin/sh

conda create -n speclet python=3.8
conda activate speclet

conda config --add channels conda-forge

conda install -c conda-forge --yes jupyterlab
conda install -c conda-forge --yes pre-commit
conda install -c conda-forge --yes jupyterlab_code_formatter
conda install -c conda-forge --yes pymc3
conda install -c conda-forge --yes matplotlib seaborn plotnine
conda install -c conda-forge --yes arviz graphviz python-graphviz
conda install -c conda-forge --yes black isort mypy nbqa watermark

pre-commit install
pre-commit run --all-files
