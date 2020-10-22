#!/bin/env Rscript

library(tidyverse)

data_file <- file.path("modeling_data", "achilles_logfold_change.csv")
out_file <- file.path("modeling_data", "all_achilles_depmapids.csv")


get_depmapids <- function(d, x) {
  dplyr::distinct(d, depmap_id)
}

read_csv_chunked(data_file, DataFrameCallback$new(get_depmapids), chunk_size = 1e5) %>%
  distinct(depmap_id) %>%
  write_csv(out_file)
