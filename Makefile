.PHONY: munge download_data docs

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_SETUP=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ;
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

help:
	@echo "available commands"
	@echo " - help               : information about available commands"
	@echo " - pyenvs             : install Python virtual environments"
	@echo " - renv               : install necessary R packages"
	@echo " - envs               : install Python and R environments"
	@echo " - download_data      : download data for the project"
	@echo " - munge              : prepare the data for analysis"
	@echo " - munge_o2           : prepare the data for analysis on O2"
	@echo " - test               : run tests"
	@echo " - test_o2            : run tests on O2 (-m 'not plots')"
	@echo " - style              : style R and Python files"
	@echo " - model_desc         : make a document of SpecletModel descriptions"
	@echo " - docs               : build documentation for Python modules"
	@echo " - clean              : remove old logs and temp files (+ style)"
	@echo " - sbc                : run the SBC pipeline (on O2)"
	@echo " - fit                : run the fitting pipeline (on O2)"
	@echo " - check_model_config : check the model configuration file"
	@echo " - build              : build the entire project"
	@echo " - build_o2           : build the entire project (on O2)"

pyenvs:
	@echo "Installing mamba in the base conda env."
	($(CONDA_SETUP) conda install -n base -c conda-forge mamba)
	@echo "Installing speclet conda environment."
	($(CONDA_SETUP) mamba env create -f environment.yaml)
	@echo "Installing snakemake conda environment."
	($(CONDA_SETUP) mamba env create -f environment_smk.yaml)

renv:
	@echo "Preparing R environment."
	Rscript -e "install.packages('renv'); renv::restore()"

envs: pyenvs renv

download_data:
	./data/download-data.sh

munge:
	./munge/munge.sh

munge_o2:
	sbatch munge/munge.sh

test:
	($(CONDA_ACTIVATE) speclet ; pytest -m "slow" & pytest -m "not slow" & wait)

test_o2:
	($(CONDA_ACTIVATE) speclet ; pytest -m "slow and not plots" & pytest -m "not slow and not plots" & wait)

test_modeling_data:
	($(CONDA_ACTIVATE) speclet ; DATA_TESTS="yes" pytest tests/test_data.py

style:
	Rscript -e "styler::style_dir('data', recursive = FALSE)"
	Rscript -e "styler::style_dir('munge', recursive = FALSE)"
	Rscript -e "styler::style_dir('.', recursive = FALSE)"
	($(CONDA_ACTIVATE) speclet && isort --profile=black speclet && isort --profile=black tests)
	($(CONDA_ACTIVATE) speclet && black speclet && black tests)
	($(CONDA_ACTIVATE) speclet && flake8 speclet && flake8 tests)

docs:
	pdoc --html -o docs --force -c latex_math=True speclet

clean: style
	find ./logs/*.log -mtime +7 | xargs rm || echo "No logs to remove.";
	find ./temp/* -mtime +7 | xargs rm -r || echo "No temp files to remove.";
	coverage erase

fit:
	sbatch pipelines/010_012_run-model-fitting-pipeline.sh

sbc:
	sbatch pipelines/012_012_run-simulation-based-calibration.sh

check_model_config:
	$(CONDA_ACTIVATE) speclet && ./speclet/command_line_interfaces/check_model_configuration.py

build: envs download_data munge test sbc fit

build_o2: envs download_data munge_o2 test_o2 sbc fit
