# Prepare the pDNA batch read counts for the Score data set.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


extract_pdna_batch_read_counts <- function(replicate_id, p_dna_batch, counts_dir) {
  read_ct_df <- read_score_count_file(
    get_score_read_count_path(counts_dir, replicate_id)
  ) %>%
    select(sgrna, tidyselect::matches(p_dna_batch)) %>%
    add_column(pdna_batch = !!p_dna_batch) %>%
    rename(read_counts = !!p_dna_batch)
}



read_counts_dir <- snakemake@params[["raw_counts_dir"]]
replicate_map_path <- snakemake@input[["replicate_map"]]
output_file <- snakemake@output[["score_pdna"]]

read_score_replicate_map(replicate_map_path) %>%
  group_by(p_dna_batch) %>%
  slice(1) %>%
  select(replicate_id, p_dna_batch)
purrr::pmap_dfr(
  process_score_read_count_replicate,
  counts_dir = score_read_counts_dir,
) %>%
  write_csv(output_file)
