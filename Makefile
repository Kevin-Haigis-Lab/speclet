.PHONY: munge download_data

help:
	@echo "available commands"
	@echo " - download_data   : download data for the project"
	@echo " - munge           : prepare the data for analysis"
	@echo " - munge_o2        : prepare the data for analysis on O2"

download_data:
	./data/download-data.sh

munge_o2:
	sbatch munge/munge.sh

munge:
	./munge/munge.sh
