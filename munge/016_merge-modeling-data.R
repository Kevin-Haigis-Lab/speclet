#!/usr/bin/env Rscript

.libPaths("/home/jc604/R-4.0/library")

library(tidyverse)

if (basename(getwd()) == "munge") {
  setwd("..")
}

#### ---- Get snakemake variables ---- ####
segmentcn_file <- snakemake@input[["segmentcn_file"]]
genecn_file <- snakemake@input[["genecn_file"]]
rna_file <- snakemake@input[["rna_file"]]
mut_file <- snakemake@input[["mut_file"]]
logfc_file <- snakemake@input[["logfc_file"]]
sampleinfo_file <- snakemake@input[["sampleinfo_file"]]

DEPMAPID <- snakemake@wildcards[["depmapid"]]

output_file <- snakemake@output[["out_file"]]



tictoc::tic(glue::glue("Process data for {DEPMAPID}"))


#### ---- Read in data frames ---- ####

segmentcn_df <- qs::qread(segmentcn_file) %>% add_column(depmap_id = DEPMAPID)
genecn_df <- qs::qread(genecn_file) %>% add_column(depmap_id = DEPMAPID)
mut_df <- qs::qread(mut_file) %>% add_column(depmap_id = DEPMAPID)
logfc_df <- qs::qread(logfc_file) %>% add_column(depmap_id = DEPMAPID)
rna_df <- qs::qread(rna_file) %>% add_column(depmap_id = DEPMAPID)
sample_info <- qs::qread(sampleinfo_file)

achilles_guide_map_path <- file.path("modeling_data", "achilles_guide_map.csv")
achilles_guide_map <- read_csv(achilles_guide_map_path)


#### ---- Copy Number ---- ####

get_segment_mean <- function(chrom, pos) {
  d <- segmentcn_df %>%
    filter(chromosome == !!chrom) %>%
    filter(start <= pos & pos <= end)
  if (nrow(d) == 0) {
    return(NA_real_)
  } else if (nrow(d) > 1) {
    stop(glue::glue("More than one segment mean value: {chrom} - {pos}"))
  } else {
    return(d$segment_mean[[1]])
  }
}

segementmean_to_copynumber <- function(seg_mean) {
  2.0^seg_mean
}


genecn_df <- genecn_df %>% rename(
  gene_cn = copy_number,
  log2_gene_cn_p1 = log2_cn_p1
)


modeling_data <- inner_join(logfc_df, sample_info, by = "depmap_id") %>%
  inner_join(achilles_guide_map, by = "sgrna") %>%
  mutate(
    chromosome = str_extract(genome_alignment, "(?<=chr)[:alnum:]{1,2}(?=_)"),
    chrom_pos = str_extract(genome_alignment, "(?<=_)[:digit:]+(?=_)"),
    chrom_pos = as.numeric(chrom_pos),
    segment_mean = map2_dbl(chromosome, chrom_pos, get_segment_mean),
    segment_cn = segementmean_to_copynumber(segment_mean)
  ) %>%
  left_join(genecn_df, by = c("depmap_id", "hugo_symbol"))



#### ---- Mutation data ---- ####


mut_df_mod <- mut_df %>%
  select(
    depmap_id, hugo_symbol,
    variant_classification, variant_type,
    is_deleterious = isdeleterious,
    is_tcga_hotspot = istcgahotspot,
    is_cosmic_hotspot = iscosmichotspot
  ) %>%
  mutate(
    variant_classification = ifelse(
      is.na(variant_classification),
      "unknown",
      variant_classification
    ),
    variant_classification = str_to_lower(variant_classification)
  ) %>%
  group_by(depmap_id, hugo_symbol) %>%
  summarise(
    n_muts = n(),
    any_deleterious = any(is_deleterious),
    variant_classification = paste(variant_classification, collapse = ";"),
    is_deleterious = paste(as.character(is_deleterious), collapse = ";"),
    is_tcga_hotspot = paste(as.character(is_tcga_hotspot), collapse = ";"),
    is_cosmic_hotspot = paste(as.character(is_cosmic_hotspot), collapse = ";"),
  ) %>%
  ungroup()


modeling_data <- left_join(
  modeling_data,
  mut_df_mod,
  by = c("depmap_id", "hugo_symbol")
) %>%
  mutate(
    n_muts = ifelse(is.na(n_muts), 0, n_muts),
    any_deleterious = ifelse(is.na(any_deleterious), FALSE, any_deleterious)
  )



#### ---- Mutation at the guide target position ---- ####


check_for_mutation_at_loc <- function(chrom, pos) {
  d <- mut_df %>%
    filter(chromosome == !!chrom & start_position <= pos & pos <= end_position)
  return(nrow(d) > 0)
}

modeling_data <- modeling_data %>%
  mutate(mutated_at_guide_location = map2_lgl(chromosome, chrom_pos, check_for_mutation_at_loc))



#### ---- RNA expression ---- ####

modeling_data <- left_join(
  modeling_data,
  rna_df,
  by = c("depmap_id", "hugo_symbol")
)

qs::qsave(modeling_data, output_file)

tictoc::toc()
