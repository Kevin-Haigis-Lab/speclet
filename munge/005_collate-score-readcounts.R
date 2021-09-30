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

# Identify the final counts column as the only remaining column after removing all
# known columns. In a previous attempt, I tried identifying the one that contained
# "CRISPR" but this was not true for all read count files.
identify_final_counts_column_name <- function(counts_df) {
  counts_df %>%
    select(-c(depmap_id, replicate_id, hugo_symbol, sgrna_id, p_dna_counts)) %>%
    colnames() %>%
    unlist()
}

save_processed_read_counts <- function(df, out_file) {
  readr::write_csv(df, out_file, append = file.exists(out_file))
  return(NULL)
}

process_score_read_count_replicate <- function(replicate_id,
                                               depmap_id,
                                               p_dna_batch,
                                               read_count_file,
                                               out_file) {
  log4r::info(logger, glue::glue("Processing raw read counts for '{replicate_id}'"))

  if (!file.exists(read_count_file)) {
    log4r::info(logger, "No read count file found - exiting early.")
    return(NULL)
  } else {
    log4r::info(logger, glue::glue("read count path: '{read_count_file}'"))
  }

  read_ct_df <- read_score_count_file(read_count_file) %>%
    add_column(depmap_id = depmap_id, replicate_id = replicate_id) %>%
    dplyr::relocate(depmap_id, replicate_id, hugo_symbol, sgrna_id) %>%
    arrange(hugo_symbol, sgrna_id)

  log4r::info(
    logger,
    glue::glue("number of rows in read count data frame: {nrow(read_ct_df)}")
  )

  if (!p_dna_batch %in% colnames(read_ct_df)) {
    stop(glue::glue("pDNA batch column not found ({p_dna_batch})"))
  }

  read_ct_df <- read_ct_df %>%
    rename(p_dna_counts = {{ p_dna_batch }})

  final_counts_col <- identify_final_counts_column_name(read_ct_df)
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
output_path <- snakemake@output[["score_raw_readcounts"]]

x <- read_score_replicate_map(replicate_map_path) %>%
  left_join(
    map_read_count_files_to_replicate_id(score_read_counts_dir),
    by = "replicate_id"
  ) %>%
  check_no_missing_count_files(read_count_file) %>%
  arrange(replicate_id) %>%
  purrr::pwalk(
    process_score_read_count_replicate,
    counts_dir = score_read_counts_dir,
    out_file = output_path
  )
