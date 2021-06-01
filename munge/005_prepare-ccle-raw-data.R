#!/usr/bin/env Rscript

# Tidy and clean CCLE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


#### ---- Data tidying functions ---- ####

tidy_rna_expression <- function(file_in, file_out) {
  read_csv(file_in, n_max = 10) %>%
    flatten_wide_df_by_gene(values_to = "rna_expr") %>%
    filter(!is.na(rna_expr)) %>%
    write_csv(file_out)
}


tidy_gene_copynumber <- function(file_in, file_out) {
  read_csv(file_in, n_max = 10) %>%
    flatten_wide_df_by_gene(values_to = "gene_cn") %>%
    write_csv(file_out)
}


tidy_gene_mutations <- function(file_in, file_out) {
  read_csv(file_in, n_max = 1e3) %>%
    janitor::clean_names() %>%
    select(
      depmap_id = dep_map_id, hugo_symbol,
      chromosome, start_pos = start_position, end_pos = end_position,
      variant_classification, variant_type, variant_annotation,
      reference_allele, tumor_seq_allele1,
      genome_change, c_dna_change, codon_change, protein_change,
      is_deleterious, is_tcga_hotspot = is_tcg_ahotspot,
      is_cosmic_hotspot = is_cosmi_chotspot
    ) %>%
    write_csv(file_out)
}


remove_noncancerous_lineages <- function(df) {
  df %>%
    filter(!str_detect(lineage, "engineer")) %>%
    filter(lineage != "embryo")
}


tidy_sample_info <- function(file_in, file_out) {
  read_csv(file_in, col_types = c("depmap_public_comments" = "c")) %>%
    janitor::clean_names() %>%
    select(
      depmap_id = dep_map_id, cell_line_name, ccle_name, sanger_model_id,
      sex, age, source, cell_line_nnmd,
      culture_type, culture_medium, cas9_activity, primary_or_metastasis,
      lineage, lineage_subtype
    ) %>%
    mutate(
      primary_or_metastasis = str_to_lower(primary_or_metastasis),
      sex = str_to_lower(sex)
    ) %>%
    remove_noncancerous_lineages() %>%
    write_csv(file_out)
}


#### ---- Function calls ---- ####

print("Tidying RNA expression.")
tidy_rna_expression(
  file_in = snakemake@input[["rna_expr"]],
  file_out = snakemake@output[["rna_expr"]]
)

print("Tidying gene copynumber.")
tidy_gene_copynumber(
  file_in = snakemake@input[["gene_cn"]],
  file_out = snakemake@output[["gene_cn"]]
)

print("Tidying gene mutations.")
tidy_gene_mutations(
  file_in = snakemake@input[["gene_mutations"]],
  file_out = snakemake@output[["gene_mutations"]]
)

print("Tidying sample info.")
tidy_sample_info(
  file_in = snakemake@input[["sample_info"]],
  file_out = snakemake@output[["sample_info"]]
)
