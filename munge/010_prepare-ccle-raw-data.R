# Tidy and clean CCLE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


# --- Data tidying functions ---

tidy_rna_expression <- function(file_in, file_out) {
  read_csv(file_in) %>%
    flatten_wide_df_by_gene(
      values_to = "rna_expr", rename_id_col_to = depmap_id, col_names_to = hugo_symbol
    ) %>%
    filter(!is.na(rna_expr)) %>%
    write_csv(file_out)
}


tidy_gene_copynumber <- function(file_in, file_out) {
  read_csv(file_in) %>%
    flatten_wide_df_by_gene(
      values_to = "gene_cn", rename_id_col_to = depmap_id, col_names_to = hugo_symbol
    ) %>%
    write_csv(file_out)
}

tidy_segment_copynumber <- function(file_in, file_out) {
  read_csv(file_in) %>%
    janitor::clean_names() %>%
    mutate(source = janitor::make_clean_names(source)) %>%
    rename(
      depmap_id = dep_map_id,
      start_pos = start,
      end_pos = end,
      amplification_status = status
    ) %>%
    mutate(
      amplification_status = case_when(
        amplification_status == "+" ~ "amp",
        amplification_status == "-" ~ "del",
        amplification_status == "0" ~ "neutral",
        amplification_status == "U" ~ "unk",
        TRUE ~ NA_character_
      )
    ) %>%
    write_csv(file_out)
}


tidy_gene_mutations <- function(file_in, file_out) {
  read_csv(file_in) %>%
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
      sex = str_to_lower(sex),
      is_male = case_when(
        sex == "male" ~ TRUE,
        sex == "female" ~ FALSE,
        sex == "unknown" ~ NA
      )
    ) %>%
    remove_noncancerous_lineages() %>%
    write_csv(file_out)
}


# --- Function calls ---

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

print("Tidying segment copy number.")
tidy_segment_copynumber(
  file_in = snakemake@input[["segment_cn"]],
  file_out = snakemake@output[["segment_cn"]]
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
