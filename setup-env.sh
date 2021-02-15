#!/bin/sh

source /home/jc604/.bashrc

conda create -n speclet -f environment.yml
conda activate speclet

pre-commit install
