# Tidy and clean Project SCORE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


#### ---- Data tidying functions ---- ####


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

check_cleaned_read_counts <- function(counts_df) {
  if (any(is.na(counts_df))) {
    na_idx <- apply(counts_df, 1, function(x) {
      any(is.na(x))
    })
    counts_df_na <- counts_df[na_idx, ]
    print(counts_df_na)
    print(pillar::glimpse(counts_df_na))
    stop("Missing data in read counts data frame")
  }
}

tidy_read_counts <- function(counts_file,
                             guide_map_file,
                             replicate_map_file,
                             out_file) {
  replicate_map <- read_score_replicate_map(replicate_map_file)
  guide_map <- read_score_guide_map(guide_map_file)

  print(head(guide_map))
  stop("JHC")

  counts_df <- readr::read_csv(counts_file, n_max = 1e5) %>%
    left_join(replicate_map, by = c("depmap_id")) %>%
    left_join(guide_map, by = c("sgrna", "hugo_symbol"))

  check_cleaned_read_counts(counts_df)

  print(counts_df)
}

#### ---- Function calls ---- ####



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
  replicate_map_file = snakemake@input[["score_replicate_map"]],
  out_file = snakemake@output[["score_read_counts"]]
)

tidy_read_counts(
  counts_file = "data/score_21q3/Score_raw_readcounts.csv",
  guide_map_file = "data/score_21q3/Score_guide_gene_map.csv",
  replicate_map_file = "data/score_21q3/Score_replicate_map.csv",
  out_file = "temp/score_read_counts.csv"
)


# raw read counts:
#   depmap_id, hugo_symbol, sgrna, counts_final, p_dna_counts, pdna_batch

# guide map:
# > sgrna,genome_alignment,gene
# > TGCTGACGGGTGACACCCA,chr19_58353007_-,A1BG (1)
# > CGGGGGTGATCCAGGACAC,chr19_58352518_+,A1BG (1)
# > TCAATGGTCACAGTAGCGC,chr19_58352307_+,A1BG (1)
# > CTGCAGCTACCGGACCGAT,chr19_58352337_-,A1BG (1)
# > GACTTCCAGCTACGGCGCG,chr19_58351567_-,A1BG (1)
# > ATCTTATCGGAGATGAAAA,chr10_50836208_-,A1CF (29974)
# > TATAAGCTCATCCTCAAAA,chr10_50844019_+,A1CF (29974)
# > AATCGGCAGTTGTCCACAC,chr10_50836281_+,A1CF (29974)
# > ACATGGTATTGCAGTAGAC,chr10_50828260_-,A1CF (29974)


# replicate map:
#   replicate_ID, DepMap_ID, pDNA_batch


# output read counts from Achilles:
# > sgrna,replicate_id,read_counts,depmap_id,p_dna_batch,passes_qc,genome_alignment,hugo_symbol,n_sgrna_alignments
# > AAAAAAATCCAGCAATGCAG,143B-311Cas9_RepA_p6_batch3,833,ACH-001001,3,TRUE,chr10_110964620_+,SHOC2,1
# > AAAAAAATCCAGCAATGCAG,21NT-311Cas9-RepAB-p6_batch4,542,ACH-002399,4,TRUE,chr10_110964620_+,SHOC2,1
# > AAAAAAATCCAGCAATGCAG,2313287-311Cas9_RepA_p5_batch3,327,ACH-000948,3,TRUE,chr10_110964620_+,SHOC2,1
# > AAAAAAATCCAGCAATGCAG,2313287-311Cas9_RepB_p5_batch3,188,ACH-000948,3,TRUE,chr10_110964620_+,SHOC2,1
# > AAAAAAATCCAGCAATGCAG,253J-311Cas9_RepA_p5_batch3,546,ACH-000011,3,TRUE,chr10_110964620_+,SHOC2,1
