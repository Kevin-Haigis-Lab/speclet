#!/usr/bin/env Rscript

# Make a list of all cell line identifiers.

library(tidyverse)

## File paths.
data_dir <- file.path("data")

OUTPUT_PATH <- file.path(data_dir, "all-depmap-ids.csv")

ccle_data_dir <- file.path(data_dir, "ccle_21q2")
score_data_dir <- file.path(data_dir, "score_21q2")
depmap_data_dir <- file.path(data_dir, "depmap_21q2")

sample_info_path <- file.path(ccle_data_dir, "CCLE_sample_info.csv")
score_replicate_path <- file.path(score_data_dir, "SCORE_replicate_map.csv")
depmap_replicate_path <- file.path(depmap_data_dir, "Achilles_replicate_map.csv")


# For each file, extract the DepMapIDs.
all_depmap_ids <- c()
for (data_file in c(sample_info_path, score_replicate_path, depmap_replicate_path)) {
  df <- read_csv(data_file)
  stopifnot("DepMap_ID" %in% colnames(df))
  all_depmap_ids <- c(all_depmap_ids, unique(unlist(df$DepMap_ID)))
}

all_depmap_ids <- unique(unlist(all_depmap_ids))
print(paste("Number of unique IDs:", length(all_depmap_ids)))


# Write output
tibble(depmap_id = all_depmap_ids) %>%
  write_csv(OUTPUT_PATH)
print(paste("Output file:", OUTPUT_PATH))
