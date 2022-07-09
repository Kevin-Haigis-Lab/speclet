# *speclet* - Bayesian modeling of genome-wide CRISPR-Cas9 LOF screens

**Use Bayesian data analysis to create informative models of genome-wide CRISPR-Cas9 loss-of-function genetic screens.**

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyerlab](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=jupyter)](https://jupyter.org) <br>
[![project-build](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/project-build.yaml/badge.svg)](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/project-build.yaml)
[![pytest](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yaml/badge.svg)](https://github.com/Kevin-Haigis-Lab/speclet/actions/workflows/CI.yaml) <br>
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![snakefmt: black](https://img.shields.io/badge/snakefmt-black-000000.svg)](https://github.com/snakemake/snakefmt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/) <br>
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

> Many setup and running commands have been added as `make` commands.
> Run `make help` to see the options available.

## Setup

### Python virtual environments

There are two ['conda'](https://docs.conda.io/en/latest/) environments for this project: the first `speclet` for modeling and analysis, the second `speclet_smk` for the pipelines.
They can be created using the following commands.
Here, we use ['mamba'](https://github.com/mamba-org/mamba) as a drop-in replacement for 'conda' to speed up the installation process.

```bash
conda install -n base -c conda-forge mamba
mamba env create -f conda.yaml
mamba env create -f conda_smk.yaml
```

Either environment can then be used like a normal 'conda' environment.
For example, below is the command it activate the `speclet` environment.

```bash
conda activate speclet
```

Alternatively, the above commands can be accomplished using the `make pyenvs` command.

```bash
# Same as above.
make pyenvs
```

On O2, because I don't have control over the `base` conda environment, I follow the incantations below for each environment:

```bash
conda create -n speclet --yes -c conda-forge python=3.9 mamba
conda activate speclet && mamba env update --name speclet --file conda.yaml
```

In addition to that fun, there is also a problem with installing Python 3.10 on the installed version of conda, so I find I need to instead install 3.9 and then let the mamba install step update it.

### GPU

Some additions to the environment need to be made in order to use a GPU for sampling from posterior distributions with the JAX backend in PyMC.
There are instructions provided on the [JAX GitHub repo](https://github.com/google/jax#pip-installation-gpu-cuda) and the [PyMC repo](https://github.com/pymc-devs/pymc/wiki/Set-up-JAX-sampling-with-GPUs-in-PyMC-v4)
First, the `cuda` and `cudnn` libraries need to be installed.
Second, a specific distribution of `jax` should be installed.
At the time of writing, the following commands work, but I would recommend consulting the two links above if doing this again in the future.

```bash
mamba install -c nvidia "cuda>=11.1" "cudnn>=8.2"
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

These commands have been added to the Makefile under the command `make gpu`.


### R environment

The ['renv'](https://rstudio.github.io/renv/) package is used to manage the R packages.
R is only used for data processing in this project.
The environment can be setup in multiple ways.
The first is by entering R and following the prompts to install the necessary packages.
Another option is to install 'renv' and running its restore command, as shown below in the R console.

```r
install.packages("renv")
renv::restore()
```

This can simply be accomplished with the following `make` command.

```bash
make renv
```

### Confirm installation

Installation of the Python virtual environment can be confirmed by running the 'speclet' test suite.

```bash
conda activate speclet
pytest
# Alternatively
make test  # or make test_o2 if on O2 HPC
```

### Pre-commit

If you plan to work on the code in this project, I recommend install ['precommit'](/Users/admin/Developer/haigis-lab/speclet/.speclet_env) so that all git commits are first checked for various style and code features.
The package is included in the `speclet` virtual environment so you just need to run the following command once.

```bash
precommit install
```

### Configuration

#### Project configuration YAML

There are options for configuration in the ["project-config.yaml"](project-config.yaml) file.
There are controls for various constants and parameters for analyses and pipelines.
Most are intuitively named.

#### Environment variables

**There is a required ".env" file that should be configured as follows.**

```text
PROJECT_ROOT=${PWD}                                 # location of the root directory
PROJECT_CONFIG=${PROJECT_ROOT}/project-config.yaml  # location of project config file
```

An optional global environment that is used by 'speclet' is `AESARA_GCC_FLAG` to set any desired Aesara gcc/g++ flags in the pipelines.
I need to have it set so that theano uses the correct gcc and blas modules when running in pipelines on O2 (see issue [#151](https://github.com/Kevin-Haigis-Lab/speclet/issues/151) for details).

## Project organization

### Data preparation

The data is downloaded to the ["data/"](data/) directory and prepared in the ["munge/"](munge/) directory.
The prepared data is available in ["modeling_data/"](modeling_data/).
Please see the READMEs in the respective directories for more information.

All of the data can be downloaded and prepared using the following commands.

```bash
make download_data
make munge # or `make munge_o2` if on O2 HPC
```

### Notebooks

Exploratory and small-scale analyses are conducted in the ["notebooks/"](notebooks/) directory.
Subdirectories divide related notebooks.
See the README in that directory for further details.

### Python Module

All shared Python code is contained in the ["speclet/"](speclet) directory.
The installation of this directory as an editable module should be done automatically when the conda environment is created.
If this failed, the module can be installed using the following command.

```python
# Run only if the module was not automatically installed by conda.
pip install -e .
```

The modules are tested using ['pytest'](https://docs.pytest.org/en/stable/) –  see below for how to run the tests.
They also conform to the ['black'](https://github.com/psf/black) and ['isort'](https://pycqa.github.io/isort/) formatters and make heavy use of Python's type-hinting system checked by ['mypy'](http://mypy-lang.org/).
The functions are well documented using the Google documentation style and are checked by ['pydocstyle'](http://www.pydocstyle.org/en/stable/).

### Pipelines

All pipelines and associated files (e.g. configurations and runners) are in the ["pipelines/"](pipelines) directory.
Each pipeline contains an associated `bash` script and `make` command that can be used to run the pipeline (usually on O2).
See the README in the ["pipelines/"](pipelines/) directory for more information.

### Reports

Standardized reports are available in the ["reports/"](reports) directory.
Each analysis pipeline has a corresponding subdirectory in the reports directory.
These notebooks are meant as quick, standardized reports to check on the results of a pipeline.
More detailed analyses are in the ["notebooks/"](notebooks/) section.

### Presentations

Presentations that involved this project are stored in the ["presentations/"](presentations) directory.
More information is available in the README in that directory.

### Testing

Tests in the ["tests/"](tests) directory have been written against the modules in ["speclet/"](speclet) using ['pytest'](https://docs.pytest.org/en/stable/) and ['hypothesis'](https://hypothesis.readthedocs.io/en/latest/).
They can be run using the following command.

```python
# Run full test suite.
pytest
# Or run the tests in two groups simultaneously.
make test  # `test_o2` on O2 HPC
```

The coverage report can be shown by adding the `--cov="speclet"` flag.
Some tests are slow because they involve the creation of models or sampling/fitting them.
These can be skipped using the `-m "not slow"` flag.
Some tests require the ability to construct plots (using the 'matplotlib' library), but not all platforms (notably the HMS research computing cluster) provide this ability.
These tests can be skipped using the `-m "not plots"` flag.

These tests are automatically run on GitHub Actions on pushes or PRs with the `master` git branch.
The most recent results can be seen [here](https://github.com/Kevin-Haigis-Lab/speclet/actions).

## Running analyses

### Pipelines

Each individual pipeline can be run through a `bash` script or a `make`command.
See the pipelines [README](pipelines/README.md) for full details.

### Notebooks

The notebooks contain the majority of the model analysis.
They are still a work in progress and this section will be updated when a full build-system is available.

```bash
# TODO
```

### Full project build

The entire project can be installed from scratch and all analysis run with the following `make` command.

```bash
make build  # or `build_o2` on O2 HPC
```
