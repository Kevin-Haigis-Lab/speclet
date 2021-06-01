.PHONY: munge, download_data

help:
        @echo "available commands"
        @echo " - download_data   : download data for the project"
        @echo " - munge           : prepare the data for analysis (O2)"

download_data:
	./data/download-data.sh

munge:
	sbatch munge/002_prepare-modeling-data_run-snakemake.sh
