# Tidy and clean Project SCORE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


# ---- Data tidying functions ----

# ---- LFC ----

tidy_log_fold_change <- function(lfc_file,
                                 guide_map_file,
                                 replicate_map_file,
                                 out_file) {
  replicate_map <- read_score_replicate_map(replicate_map_file)
  guide_map <- read_score_guide_map(guide_map_file)

  readr::read_csv(lfc_file, n_max = 1e2) %>%
    flatten_wide_df_by_gene(
      values_to = "lfc", rename_id_col_to = sgrna, col_names_to = replicate_id
    ) %>%
    inner_join(replicate_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    readr::write_csv(out_file)
}

# ---- Read counts ----

read_tzelepis_guide_map <- function(path) {
  readr::read_csv(path)
}


check_cleaned_read_counts <- function(counts_df) {
  if (any(is.na(counts_df))) {
    na_idx <- apply(counts_df, 1, function(x) {
      any(is.na(x))
    })
    counts_df_na <- counts_df[na_idx, ]
    print(counts_df_na)
    print(pillar::glimpse(counts_df_na))
    print(apply(counts_df, 2, function(x) {
      sum(is.na(x))
    }))
    stop("Missing data in read counts data frame")
  }
}

tidy_read_counts <- function(counts_file,
                             guide_map_file,
                             tzelepis_guide_map_file,
                             out_file) {
  score_guide_map <- read_score_guide_map(guide_map_file)
  tzelepis_guide_map <- read_tzelepis_guide_map(tzelepis_guide_map_file)

  counts_df <- readr::read_csv(counts_file) %>%
    inner_join(tzelepis_guide_map, by = c("sgrna_id")) %>%
    inner_join(score_guide_map, by = c("sgrna", "hugo_symbol"))

  check_cleaned_read_counts(counts_df)
  readr::write_csv(counts_df, out_file)
}

# ---- Function calls ----

print("---- Tidying SCORE log fold change. ----")
tidy_log_fold_change(
  lfc_file = snakemake@input[["score_log_fold_change"]],
  guide_map_file = snakemake@input[["score_guide_map"]],
  replicate_map_file = snakemake@input[["score_replicate_map"]],
  out_file = snakemake@output[["log_fold_change"]]
)

print("---- Tidying SCORE read counts. ----")
tidy_read_counts(
  counts_file = snakemake@input[["score_raw_readcounts"]],
  guide_map_file = snakemake@input[["score_guide_map"]],
  tzelepis_guide_map_file = snakemake@input[["tzelepis_sgnra_lib"]],
  out_file = snakemake@output[["score_read_counts"]]
)
