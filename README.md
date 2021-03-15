# *KRAS* allele-specific synthetic lethal analysis using Bayesian statistics

**Using Bayesian statistics to model CRISPR-Cas9 genetic screen data to identify, with measurable uncertainty, synthetic lethal interactions that are specific to the individual *KRAS* mutations.**

[![python](https://img.shields.io/badge/Python-3.9.1-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![jupyerlab](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=jupyter)](https://jupyter.org) <br>
[![pytest](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yml/badge.svg)](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) <br>
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Index

### [Data preparation](munge/)

All of the preparation of source data was conducted in the [Munge](munge/) directory. See the README in that directory for further details.

### [Analysis](analysis/)

All of the analysis was conducted in the [Analysis](analysis/) directory.
See the README in that directory for further details.

### [Testing](tests/)

Tests have been written against the modules in `analysis/`.
They can be run using the following command.

```python
python3 -m pytest --disable-warnings tests/
```

The coverage report can be shown by adding the `--cov="analysis"` parameter.
Some tests are slow because they involve the creation of PyMC3 models or sampling/fitting them.
These can be skipped using the `-m "not slow"` argument.

These tests are automatically run on GitHub Actions on pushes or PRs with the `master` git branch.
The most recent results can be seen [here](https://github.com/Kevin-Haigis-Lab/speclet/actions).
