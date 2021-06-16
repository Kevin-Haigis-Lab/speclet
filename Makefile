.PHONY: munge download_data docs

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_SETUP=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ;
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

help:
	@echo "available commands"
	@echo " - help            : information about available commands"
	@echo " - install         : install virtual environments (Python and R)"
	@echo " - download_data   : download data for the project"
	@echo " - munge           : prepare the data for analysis"
	@echo " - munge_o2        : prepare the data for analysis on O2"
	@echo " - test            : run tests"
	@echo " - test_o2         : run tests on O2 (-m 'not plots')"
	@echo " - style           : style R and Python files"
	@echo " - docs            : build documentation for Python modules"
	@echo " - clean           : remove old logs and temp files"
	@echo " - sbc             : run the SBC pipeline (on O2)"
	@echo " - fit             : run the fitting pipeline (on O2)"

install:
	@echo "Installing speclet conda environment."
	($(CONDA_SETUP) conda env create -f environment.yml)
	@echo "Installing snakemake conda environment."
	($(CONDA_SETUP) conda env create -f snakemake_environment.yml)
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

style:
	Rscript -e "styler::style_dir()"
	($(CONDA_ACTIVATE) speclet && isort src && isort tests)
	($(CONDA_ACTIVATE) speclet && black src && black tests)
	($(CONDA_ACTIVATE) speclet && flake8 src && flake8 tests)

docs:
	pdoc --html -o docs --force -c latex_math=True src

clean:
	find ./logs/*.log -mtime +7 -exec rm {} \ || echo "No logs to remove.";
	find ./temp/* -mtime +7 -exec rm {} \ || echo "No temp files to remove.";
	rm .coverage* || echo "No coverage remnants to remove."

sbc:
	sbatch pipelines/012_012_simulation-based-calibration.sh

fit:
	sbatch pipelines/010_012_run-crc-sampling.sh
