# Collate the raw read counts data from Project SCORE into a single file.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(glue)
library(log4r)
library(tidyverse)

source("munge/munge_functions.R")


logger <- logger("INFO")


read_score_replicate_map <- function(path) {
  readr::read_csv(path) %>%
    janitor::clean_names() %>%
    dplyr::rename(depmap_id = dep_map_id)
}

read_score_count_file <- function(path) {
  readr::read_tsv(path) %>%
    rename(sgrna = sgRNA, hugo_symbol = gene)
}

assert_only_one_possible_final_count_column <- function(counts_df, possible_cols) {
  if (length(possible_cols) == 1) {
    return(NULL)
  } else if (length(possible_cols) == 0) {
    print(head(counts_df), n = 5)
    stop("Could not find final counts column.")
  } else if (length(possible_cols) > 1) {
    print(head(counts_df), n = 5)
    cols <- paste(possible_cols, collapse = ", ") # nolint
    print(glue::glue("possible columns: {cols}"))
    stop("Found multiple columns for the final count.")
  } else {
    stop("Unforseen error.")
  }
}

save_processed_read_counts <- function(df, out_file) {
  readr::write_csv(df, out_file, append = file.exists(out_file))
  return(NULL)
}

process_score_read_count_replicate <- function(replicate_id,
                                               depmap_id,
                                               p_dna_batch,
                                               counts_dir,
                                               out_file) {
  log4r::info(logger, glue::glue("Processing raw read counts for '{replicate_id}'"))
  read_ct_file <- paste0(replicate_id, ".read_count.tsv.gz")
  read_ct_path <- file.path(counts_dir, read_ct_file)
  log4r::info(logger, glue::glue("read count path: '{read_ct_path}'"))
  read_ct_df <- read_score_count_file(read_ct_path) %>%
    add_column(depmap_id = depmap_id) %>%
    dplyr::relocate(depmap_id, hugo_symbol, sgrna) %>%
    arrange(hugo_symbol, sgrna)

  log4r::info(
    logger,
    glue::glue("number of rows in read count data frame: {nrow(read_ct_path)}")
  )

  if (!p_dna_batch %in% colnames(read_ct_df)) {
    stop(glue::glue("pDNA batch column not found ({p_dna_batch})"))
  }

  read_ct_df <- read_ct_df %>%
    rename(p_dna_counts = {{ p_dna_batch }})

  crispr_col_idx <- stringr::str_detect(colnames(read_ct_df), "CRISPR")
  final_counts_col <- colnames(read_ct_df)[crispr_col_idx]
  final_counts_col <- unlist(final_counts_col)
  assert_only_one_possible_final_count_column(read_ct_df, final_counts_col)
  final_counts_col <- final_counts_col[[1]]
  log4r::info(logger, glue::glue("final count column: '{final_counts_col}'"))

  read_ct_df <- read_ct_df %>%
    rename(counts_final = {{ final_counts_col }}) %>%
    add_column(pdna_batch = !!p_dna_batch)

  save_processed_read_counts(read_ct_df, out_file)
  return(NULL)
}


score_read_counts_dir <- snakemake@params[["raw_counts_dir"]]
replicate_map_path <- snakemake@input[["replicate_map"]]
output_file <- snakemake@output[["score_raw_readcounts"]]

## For testing
# score_read_counts_dir <- "data/score_21q3/Score_raw_sgrna_counts/SecondBatch"
# replicate_map_path <- "data/score_21q3/Score_replicate_map.csv"
# output_path <- "temp/Score_raw_readcounts.csv"

x <- read_score_replicate_map(replicate_map_path) %>%
  arrange(replicate_id) %>%
  purrr::pwalk(
    process_score_read_count_replicate,
    counts_dir = score_read_counts_dir,
    out_file = output_path
  )
