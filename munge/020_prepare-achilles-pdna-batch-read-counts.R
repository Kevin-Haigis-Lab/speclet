# Prepare the pDNA batch read counts for the Achilles data set.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")

# Steps followed from "Extracting Biological Insights from the Project Achilles Genome-
# Scale CRISPR Screens in Cancer Cell Lines."
#
# 1. "remove suspected off-target Avana sgRNAs"
# 2. "set to null (NA) pDNA measurements for sgRNAs if the measurement is less than one
#    read per million for that pDNA batch"
#   a)  "sgRNA reads are then NAed in all replicates belonging to the same pDNA batch."
# 3. "remove replicates that have fewer than 15 million total read counts"
# 4. "normalize each read count column by its total reads, multiply by one million, and
#    add one pseudocount to generate reads per million (RPM)"
#

coalesce_sgrna_initial_reads <- function(reads_df) {
  reads_df %>%
    group_by(replicate_id) %>%
    mutate(
      rpm = (reads / sum(reads)) * 1e6,
    ) %>%
    ungroup() %>%
    group_by(p_dna_batch, hugo_symbol, sgrna) %>%
    summarize(median_rpm = median(rpm)) %>%
    ungroup() %>%
    mutate(
      less_that_1_rpm = median_rpm < 1.0,
      median_rpm = median_rpm + 1
    )
}


read_readcounts_data <- function(f, keep) {
  data.table::fread(f, select = keep) %>%
    as_tibble() %>%
    rename(sgrna = "Construct Barcode")
}



prepare_achilles_pdna_batch_read_counts <- function(guide_map_file,
                                                    dropped_guides_file,
                                                    replicate_map_file,
                                                    achilles_read_counts_file,
                                                    out_file) {
  guide_map <- read_achilles_guide_map(guide_map_file) %>%
    select(sgrna, hugo_symbol)

  pdna_replicate_map <- read_achilles_replicate_map(replicate_map_file) %>%
    filter(is.na(depmap_id)) %>%
    filter(passes_qc) %>%
    select(replicate_id, p_dna_batch) %>%
    distinct() %>%
    arrange(p_dna_batch, replicate_id)

  dropped_guides <- readr::read_csv(dropped_guides_file)$X1
  keep_cols <- c("Construct Barcode", pdna_replicate_map$replicate_id)

  read_counts <- read_readcounts_data(achilles_read_counts_file, keep = keep_cols) %>%
    filter(!(sgrna %in% dropped_guides)) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "reads") %>%
    left_join(pdna_replicate_map, by = c("replicate_id")) %>%
    left_join(guide_map, by = "sgrna")

  return(read_counts)
}



pdna_read_counts <- prepare_achilles_pdna_batch_read_counts(
  guide_map_file = snakemake@input[["guide_map"]],
  dropped_guides_file = snakemake@input[["dropped_guides"]],
  replicate_map_file = snakemake@input[["replicate_map"]],
  achilles_read_counts_file = snakemake@input[["achilles_read_counts"]]
)

# pDNA read counts for each replicate sequencing within a pDNA batch.
readr::write_csv(pdna_read_counts, snakemake@output[["achilles_replcate_pdna_counts"]])

# Coalesced pDNA read counts for each batch. Single pDNA initial count per batch.
coalesce_sgrna_initial_reads(pdna_read_counts) %>%
  readr::write_csv(snakemake@output[["achilles_batch_pdna_counts"]])
