#!/bin/sh

conda create -n speclet -f environment.yml
conda activate speclet

pre-commit install
