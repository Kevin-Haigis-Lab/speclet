# *speclet* - Bayesian modeling of genome-wide CRISPR-Cas9 LOF screens

**Use Bayesian data analysis to create flexible and informative models of genome-wide CRISPR-Cas9 loss-of-function genetic screens.**

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![jupyerlab](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=jupyter)](https://jupyter.org) <br>
[![project-build](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/project-build.yaml/badge.svg)](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/project-build.yaml)
[![pytest](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yaml/badge.svg)](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yaml)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: snakefmt](https://img.shields.io/badge/code%20style-snakefmt-000000.svg)](https://github.com/snakemake/snakefmt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/) <br>
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Index

### [Data preparation](munge/)

The data is downloaded to the ["data/"](data) directory and prepared in the ["munge/"](munge) directory.
The prepared data is available in ["modeling_data/"](modeling_data).

All of the data can be downloaded and prepared using the following command.

```bash
make download_data
make munge # or `make munge_o2` if on O2 HPC
```

### [Notebooks](notebooks/)

Exploratory and small-scale analyses are conducted in the "notebooks/" directory.
Subdirectories divide related notebooks.
See the README in that directory for further details.

### [Python Modules](src/)

All shared Python code is contained in the ["src/"](src) directory.
The installation of this directory as an editable module should be done automatically when the conda environment is created.
If this failed, the module can be installed using the following command.

```python
# Run only if the module was not automatically installed by conda.
pip install -e .
```

The modules are tested using ['pytest'](https://docs.pytest.org/en/stable/) –  see below for how to run the tests.
They also conform to the ['black'](https://github.com/psf/black) formatter and make heavy use of Python's type-hinting system checked by ['mypy'](http://mypy-lang.org/).
The functions are well documented using the Google documentation style and are checked by ['pydocstyle'](http://www.pydocstyle.org/en/stable/).

### [Pipelines](pipelines/)

All pipelines and associated files (e.g. configurations and runners) are in the ["pipelines/"](pipelines) directory.
Each pipeline should contain an associated bash script and `make` command that can be used to run the pipeline (usually on O2).

### [Reports](reports/)

Standardized reports are available in the ["reports/"](reports) directory.
Each annalysis pipeline should have a corresponding subdirectory in the reports directory.

### [Presentations](presentations/)

Presentations that involved this project are stored in the ["presentations/"](presentations) directory.

### [Testing](tests/)

Tests in the ["tests/"](tests) directory have been written against the modules in ["src/"](src) using ['pytest'](https://docs.pytest.org/en/stable/) and ['hypothesis'](https://hypothesis.readthedocs.io/en/latest/).
They can be run using the following command.

```python
# Run full test suite.
pytest
# Or run the tests in two groups simultaneously.
make test  # `test_o2` on O2
```

The coverage report can be shown by adding the `--cov="src"` flag.
Some tests are slow because they involve the creation of PyMC3 models or sampling/fitting them.
These can be skipped using the `-m "not slow"` argument.
Some tests require the ability to construct plots (using the 'matplotlib' library), but not all platforms (notably the HMS research computing cluster) provide this ability.
These tests can be skipped using the `-m "not plots"` argument.

These tests are automatically run on GitHub Actions on pushes or PRs with the `master` git branch.
The most recent results can be seen [here](https://github.com/Kevin-Haigis-Lab/speclet/actions).
