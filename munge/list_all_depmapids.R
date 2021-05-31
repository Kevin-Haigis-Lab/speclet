#!/usr/bin/env Rscript

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(tidyverse)

data_file <- snakemake@input[["data_file"]] # "achilles_logfold_change.csv"
out_file <- snakemake@input[["out_file"]] # "all_achilles_depmapids.csv"

get_depmapids <- function(d, x) {
  dplyr::distinct(d, depmap_id)
}

read_csv_chunked(data_file, DataFrameCallback$new(get_depmapids), chunk_size = 1e5) %>%
  distinct(depmap_id) %>%
  write_csv(out_file)
