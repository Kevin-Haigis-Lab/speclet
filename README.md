# *KRAS* allele-specific synthetic lethal analysis using Bayesian statistics

**Using Bayesian statistics to model CRISPR-Cas9 genetic screen data to identify, with measureable uncertainty, synthetic lethal interactions that are specific to the individual *KRAS* mutations.**

[![python](https://img.shields.io/badge/Python-3.7.4-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![jupyerlab](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=jupyter)](https://jupyter.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

---

## Index

### [Data preparation](munge/)

1. [Prepare DepMap data](munge/005_prepare-depmap-data.md)
2. [Prepare modeling data](munge/010_prepare-modeling-data.md)
3. [Subsample the modeling data](munge/019_prepare-data-subsample.md)

### [Analysis](analysis/)

1. [Experiment with model designs](analysis/)

---

## To-Do

- design increasingly complex models
- design a processing workflow for the models
    - should be consistent yet customizable to the needs of different models
    - decide on diagnositic values and plots to collect and a system of documentation
