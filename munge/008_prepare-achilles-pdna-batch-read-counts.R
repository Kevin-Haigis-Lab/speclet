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

reads_per_million <- function(df) {
  reads_df %>%
    group_by(replicate_id) %>%
    mutate(rpm = (reads / sum(reads)) * 1e6) %>%
    ungroup()
}

read_readcounts_data <- function(f) {
  data.table::fread(achilles_read_counts_file, select = keep_cols) %>%
    as_tibble() %>%
    rename(sgrna = "Construct Barcode")
}



prepare_achilles_pdna_batch_read_counts <- function(guide_map_file,
                                                    dropped_guides_file,
                                                    replicate_map_file,
                                                    achilles_read_counts_file,
                                                    out_file) {
  guide_map <- read_guide_map(guide_map_file) %>%
    select(sgrna, hugo_symbol)

  pdna_replicate_map <- read_replicate_map(replicate_map_file) %>%
    filter(is.na(depmap_id)) %>%
    filter(passes_qc) %>%
    select(replicate_id, p_dna_batch) %>%
    distinct() %>%
    arrange(p_dna_batch, replicate_id)

  dropped_guides <- readr::read_csv(dropped_guides_file)$X1

  keep_cols <- c("Construct Barcode", pdna_replicate_map$replicate_id)

  read_counts <- read_readcounts_data(achilles_read_counts_file) %>%
    filter(!(sgrna %in% dropped_guides)) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "reads") %>%
    left_join(pdna_replicate_map, by = c("replicate_id")) %>%
    left_join(guide_map, by = "sgrna") %>%
    reads_per_million()

  readr::write_csv(read_counts, out_file)
}



prepare_achilles_pdna_batch_read_counts(
  guide_map_file = snakemake@input[["guide_map"]],
  dropped_guides_file = snakemake@input[["dropped_guides"]],
  replicate_map_file = snakemake@input[["replicate_map"]],
  achilles_read_counts_file = snakemake@input[["achilles_read_counts"]],
  out_file = snakemake@output[["achilles_batch_pdna_counts"]]
)
# prepare_achilles_pdna_batch_read_counts(
#   guide_map_file = "data/depmap_21q3/Achilles_guide_map.csv",
#   dropped_guides_file = "data/depmap_21q3/Achilles_dropped_guides.csv",
#   replicate_map_file = "data/depmap_21q3/Achilles_replicate_map.csv",
#   achilles_read_counts_file = "data/depmap_21q3/Achilles_raw_readcounts.csv",
#   out_file = "temp/achilles_pdna_example.csv"
# )
