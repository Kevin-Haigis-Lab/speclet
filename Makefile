.PHONY: munge download_data docs

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_SETUP=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ;
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

help:
	@echo "available commands"
	@echo " - help               : information about available commands"
	@echo " - envs               : install Python virtual environments"
	@echo " - install            : install Python and R virtual environments"
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

envs:
	@echo "Installing speclet conda environment."
	($(CONDA_SETUP) conda env create -f environment.yaml)
	@echo "Installing snakemake conda environment."
	($(CONDA_SETUP) conda env create -f snakemake_environment.yaml)

install: envs
	@echo "Preparing R environment."
	Rscript -e "renv::restore()"

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
	($(CONDA_ACTIVATE) speclet && isort --profile=black src && isort --profile=black tests)
	($(CONDA_ACTIVATE) speclet && black src && black tests)
	($(CONDA_ACTIVATE) speclet && flake8 src && flake8 tests)

model_desc:
	python3 src/command_line_interfaces/speclet_cli.py model-docs

docs: model_desc
	pdoc --html -o docs --force -c latex_math=True src

clean: style
	find ./logs/*.log -mtime +7 | xargs rm || echo "No logs to remove.";
	find ./temp/* -mtime +7 | xargs rm -r || echo "No temp files to remove.";
	coverage erase

sbc:
	sbatch pipelines/012_012_simulation-based-calibration.sh

fit:
	sbatch pipelines/010_012_run-crc-sampling.sh

check_model_config:
	$(CONDA_ACTIVATE) speclet && ./src/command_line_interfaces/check_model_configuration.py models/model-configs.yaml

build: install download_data munge test sbc fit

build_o2: install download_data munge_o2 test_o2 sbc fit
