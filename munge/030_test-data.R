#!/usr/bin/env Rscript

# Prepare data to use is tests.

library(magrittr)
library(tidyverse)

data_dir <- here::here("modeling_data")
data <- data.table::fread(
  file.path(data_dir, "depmap_modeling_dataframe.csv"),
  showProgress = FALSE,
  nrows = 2e6
)
data <- as_tibble(data)

# Shuffle the data - order should NOT matter!
data <- slice_sample(data, n = 1e3)

write_csv(data, here::here("tests", "depmap_test_data.csv"))
