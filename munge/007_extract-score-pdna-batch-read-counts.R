# Prepare the pDNA batch read counts for the Score data set.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")



extract_pdna_batch_read_counts <- function(replicate_id, p_dna_batch, read_count_file) {
  read_ct_df <- read_score_count_file(read_count_file) %>%
    select(sgrna_id, tidyselect::matches(p_dna_batch)) %>%
    add_column(pdna_batch = !!p_dna_batch) %>%
    rename(read_counts = !!p_dna_batch)
  return(read_ct_df)
}


score_read_counts_dir <- snakemake@params[["raw_counts_dir"]]
replicate_map_path <- snakemake@input[["replicate_map"]]
output_file <- snakemake@output[["score_pdna"]]

# Select a single replicate for each pDNA batch and extract the pDNA reads from
# that counts file.
read_score_replicate_map(replicate_map_path) %>%
  left_join(
    map_read_count_files_to_replicate_id(score_read_counts_dir),
    by = "replicate_id"
  ) %>%
  check_no_missing_count_files(read_count_file) %>%
  group_by(p_dna_batch) %>%
  slice(1) %>%
  ungroup() %>%
  select(replicate_id, p_dna_batch, read_count_file) %>%
  purrr::pmap_dfr(
    extract_pdna_batch_read_counts
  ) %>%
  write_csv(output_file)
