#!/usr/bin/env Rscript

# Prepare data to use is tests.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)

data_dir <- here::here("modeling_data")
data <- data.table::fread(
  snakemake@input[["depmap_modeling_df"]],
  showProgress = FALSE,
  nrows = 2e6
)
data <- as_tibble(data)

# Shuffle the data - order should NOT matter!
data <- slice_sample(data, n = 1e3)

write_csv(data, snakemake@output[["output_dest"]])
