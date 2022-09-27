# Prepare cancer gene list from Bailey et al., Cell, 2018.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(glue)
library(tidyverse)


# --- Snakemake interfacing ---

# TODO: add to munge pipeline
# Input
# bailey_supp_excel_file <- snakemake@input[["bailey_supp_excel"]]
# cgc_genes_df_file <- snakemake@input[["cgc_genes"]]
# depmap_cell_line_info_file <- snakemake@input[["cell_line_info"]]
# depmap_cancer_types_file <- snakemake@input[["depmap_cancer_map"]]

# # Output
# bailey_genes_df_file <- snakemake@output[["bailey_genes_df"]]
# bailey_gene_map_file <- snakemake@output[["bailey_genes_dict"]]
# cgc_gene_map_file <- snakemake@output[["cgc_genes_dict"]]

# TODO: remove when finished dev
bailey_supp_excel_file <- "data/bailey-2018-cell/bailey-cancer-genes.xlsx"
cgc_genes_df_file <- "modeling_data/sanger_cancer-gene-census.csv"
depmap_cell_line_info_file <- "modeling_data/depmap_cell-line-info.csv"
depmap_cancer_types_file <- "data/depmap-lineage-cancer-types.tsv"

bailey_genes_df_file <- "modeling_data/bailey-cancer-genes.csv"
bailey_gene_map_file <- "modeling_data/bailey-cancer-genes-dict.json"
cgc_gene_map_file <- "modeling_data/cgc-cancer-genes-dict.json"


# --- DepMap lineages ---

depmap_cell_line_info <- read_csv(depmap_cell_line_info_file)
depmap_cancer_types <- read_tsv(depmap_cancer_types_file)

# Check all cancers accounted for.
missing_cancers <- depmap_cell_line_info %>%
  distinct(lineage, lineage_subtype) %>%
  dplyr::setdiff(depmap_cancer_types %>% distinct(lineage, lineage_subtype))

stopifnot(nrow(missing_cancers) == 0)


# --- Prepare Bailey cancer genes ---

bailey_genes_df <- readxl::read_excel(
  bailey_supp_excel_file,
  sheet = "Table S1",
  skip = 3
) %>%
  janitor::clean_names() %>%
  select(
    hugo_symbol = gene,
    cancer,
    tsg_or_oncogene = tumor_suppressor_or_oncogene_prediction_by_20_20
  ) %>%
  mutate(role = case_when(
    is.na(tsg_or_oncogene) ~ NA_character_,
    str_detect(tsg_or_oncogene, "oncogene") ~ "oncogene",
    str_detect(tsg_or_oncogene, "tsg") ~ "tsg",
    TRUE ~ NA_character_,
  )) %>%
  select(-tsg_or_oncogene) %>%
  distinct()

write_csv(bailey_genes_df, bailey_genes_df_file)


# --- Prepare CGC cancer genes ---

cgc_genes_df <- read_csv(cgc_genes_df_file) %>%
  filter(tier == 1) %>%
  filter(is_oncogene | is_tsg) %>%
  select(hugo_symbol, tumor_types_somatic) %>%
  mutate(tumor_types_somatic = str_split(tumor_types_somatic, ";")) %>%
  unnest(tumor_types_somatic) %>%
  mutate(tumor_types_somatic = str_trim(tumor_types_somatic)) %>%
  rename(cancer = tumor_types_somatic)


# --- Mapping cancer genes to each DepMap lineage and sub-lineage ---

collect_cancer_genes <- function(cancer_codes_str, cancer_genes_df, pancancer = NULL) {
  cancer_codes <- unlist(str_split(cancer_codes_str, ","))
  cancer_codes <- c(cancer_codes, pancancer)
  genes <- cancer_genes_df %>%
    filter(cancer %in% cancer_codes) %>%
    pull(hugo_symbol) %>%
    unlist() %>%
    unique()
  return(genes)
}

depmap_cancer_genes <- depmap_cancer_types %>%
  mutate(
    bailey_genes = purrr::map(
      tcga_code, ~ collect_cancer_genes(.x, bailey_genes_df, pancancer = "PANCAN")
    ),
    cgc_genes = purrr::map(cgc_code, ~ collect_cancer_genes(.x, cgc_genes_df))
  )


# --- Write out to JSON ---

make_cancer_gene_list <- function(gene_df, gene_col) {
  lineages <- sort(unique(unlist(gene_df$lineage)))
  gene_list <- rep(NA, length(lineages))
  names(gene_list) <- lineages

  for (line in lineages) {
    line_df <- gene_df %>%
      filter(lineage == !!line)
    genes <- line_df %>% pull({{ gene_col }})
    names(genes) <- unlist(line_df$lineage_subtype)
    gene_list[line] <- list(genes)
  }

  return(gene_list)
}

depmap_gene_list_bailey <- make_cancer_gene_list(depmap_cancer_genes, bailey_genes)
depmap_gene_list_cgc <- make_cancer_gene_list(depmap_cancer_genes, cgc_genes)

write(rjson::toJSON(depmap_gene_list_bailey), bailey_gene_map_file)
write(rjson::toJSON(depmap_gene_list_cgc), cgc_gene_map_file)
