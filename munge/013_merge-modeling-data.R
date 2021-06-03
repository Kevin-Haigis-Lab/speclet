#!/usr/bin/env Rscript

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(log4r)
library(glue)
library(tidyverse)

logger <- logger("DEBUG")

#### ---- Interactions with Snakemake ---- ####

ccle_rna_file <- snakemake@input[["ccle_rna"]]
ccle_gene_cn_file <- snakemake@input[["ccle_gene_cn"]]
ccle_segment_cn_file <- snakemake@input[["ccle_segment_cn"]]
ccle_mut_file <- snakemake@input[["ccle_mut"]]
achilles_lfc_file <- snakemake@input[["achilles_lfc"]]
score_cn_file <- snakemake@input[["score_cn"]]
score_lfc_file <- snakemake@input[["score_lfc"]]
sample_info_file <- snakemake@input[["sample_info"]]

out_file <- snakemake@output[["out_file"]]

# ccle_rna_file <- "temp/ccle-rna_ACH-000001.qs"
# ccle_gene_cn_file <- "temp/ccle-genecn_ACH-000001.qs"
# ccle_segment_cn_file <- "temp/ccle-segmentcn_ACH-000001.qs"
# ccle_mut_file <- "temp/ccle-mut_ACH-000001.qs"
# achilles_lfc_file <- "temp/achilles-lfc_ACH-000001.qs"
# score_cn_file <- "temp/score-segmentcn_ACH-000001.qs"
# score_lfc_file <- "temp/score-lfc_ACH-000001.qs"
# sample_info_file <- "modeling_data/ccle_sample_info.csv"


#### ---- Data retrieval functions ---- ####

prepare_lfc_data <- function(lfc_df, screen_scoure) {
  lfc_df %>%
    add_column(screen = screen_scoure) %>%
    filter(!is.na(hugo_symbol)) %>%
    distinct()
}

get_log_fold_change_data <- function(achilles_lfc_path, score_lfc_path) {
  info(logger, glue("Retrieving LFC data ({achilles_lfc_path}, {score_lfc_path})."))

  achilles_lfc <- qs::qread(achilles_lfc_path) %>%
    prepare_lfc_data(screen_scoure = "broad") %>%
    select(-n_sgrna_alignments) %>%
    mutate(p_dna_batch = as.character(p_dna_batch))

  score_lfc <- qs::qread(score_lfc_path) %>%
    prepare_lfc_data(screen_scoure = "sanger")

  if (nrow(achilles_lfc) == 0 && nrow(score_lfc) == 0) {
    msg <- "LFC data not found from either data source."
    warn(logger, msg)
    return(NULL)
  } else if (nrow(achilles_lfc) == 0) {
    info(logger, "Data found from Project SCORE.")
    return(score_lfc)
  } else if (nrow(score_lfc) == 0) {
    info(logger, "Data found from Achilles.")
    return(achilles_lfc)
  } else {
    info(logger, "Data found from both projects.")
    return(bind_rows(achilles_lfc, score_lfc))
  }
}

extract_depmap_id <- function(rna_f) {
  info(logger, glue("Extracting DepMap ID from '{rna_f}'"))
  id <- basename(rna_f) %>%
    tools::file_path_sans_ext() %>%
    str_split("_") %>%
    unlist()
  id <- id[[2]]
  info(logger, glue("Found '{id}'"))
  return(id)
}

get_sample_info <- function(sample_info_path, depmap_id) {
  info(logger, "Retrieving sample information.")
  sample_info <- read_csv(sample_info_path) %>%
    filter(depmap_id == !!depmap_id)

  if (nrow(sample_info) == 0) {
    msg <- glue("Did not find sample information for DepMap ID '{depmap_id}'.")
    error(logger, msg)
    stop(msg)
  } else if (nrow(sample_info) > 1) {
    print(sample_info)
    msg <- glue("Found {nrow(sample_info)} samples with DepMap ID '{depmap_id}'.")
    error(logger, msg)
    stop(msg)
  }

  return(sample_info)
}

remove_sgrna_that_target_multiple_genes <- function(lfc_df) {
  if (is.null(lfc_df)) {
    return(lfc_df)
  }

  target_counts <- lfc_df %>%
    distinct(sgrna, hugo_symbol) %>%
    count(sgrna) %>%
    filter(n != 1)
  multi_target <- unique(unlist(target_counts$sgrna))

  warn(
    logger,
    glue("Removing {length(multi_target)} sgRNAs with multiple targets.")
  )
  lfc_df %>% filter(!(sgrna %in% !!multi_target))
}

get_rna_expression_data <- function(rna_path, filter_genes = NULL) {
  info(logger, glue("Retrieving RNA expression data ({rna_path})."))
  rna <- qs::qread(rna_path) %>%
    distinct()

  if (!is.null(filter_genes)) {
    info(logger, "Filtering RNA expression data for specific genes.")
    rna <- rna %>% filter(hugo_symbol %in% filter_genes)
  }

  return(rna)
}

get_mutation_data <- function(mut_path, filter_genes = NULL) {
  info(logger, glue("Retrieving mutation data ({mut_path})."))
  mut <- qs::qread(mut_path) %>%
    distinct() %>%
    filter(!variant_annotation %in% c("silent")) %>%
    group_by(hugo_symbol) %>%
    summarise(
      num_mutations = n(),
      any_deleterious = any(is_deleterious),
      any_tcga_hotspot = any(is_tcga_hotspot),
      any_cosmic_hotspot = any(is_cosmic_hotspot)
    ) %>%
    add_column(is_mutated = TRUE)

  if (!is.null(filter_genes)) {
    info(logger, "Filtering mutation data for specific genes.")
    mut <- mut %>% filter(hugo_symbol %in% filter_genes)
  }

  return(mut)
}

get_gene_copy_number_data <- function(ccle_gene_cn_path, filter_genes = NULL) {
  info(logger, glue("Retrieving gene CN data ({ccle_gene_cn_path})."))
  cn <- qs::qread(ccle_gene_cn_path) %>%
    distinct()

  if (!is.null(filter_genes)) {
    info(logger, "Filtering gene CN data for specific genes.")
    cn <- cn %>% filter(hugo_symbol %in% filter_genes)
  }

  return(cn)
}

get_segment_copy_number_data <- function(ccle_segment_cn_path,
                                         sanger_segment_cn_path) {
  info(
    logger,
    glue("Retrieving segment CN data ({ccle_segment_cn_path}, {sanger_segment_cn_path}).")
  )
  ccle_cn <- qs::qread(ccle_segment_cn_path) %>% distinct()
  sanger_cn <- qs::qread(sanger_segment_cn_path) %>% distinct()
  if (nrow(ccle_cn) == 0 && nrow(ccle_cn) == 0) {
    msg <- "Segment CN data not found from either CCLE nor Sanger."
    erorr(logger, msg)
    stop(msg)
  } else if (nrow(ccle_cn) == 0) {
    info(logger, "Segment CN found from Sanger.")
    return(ccle_cn)
  } else if (nrow(ccle_cn) == 0) {
    info(logger, "Segment CN found from CCLE.")
    return(ccle_cn)
  } else {
    info(logger, "Data found from both CCLE and Sanger.")
    cn <- bind_rows(ccle_cn, ccle_cn) %>% distinct()
    return(cn)
  }
}


lfc_data <- get_log_fold_change_data(achilles_lfc_file, score_lfc_file) %>%
  remove_sgrna_that_target_multiple_genes()

if (is.null(lfc_data)) {
  warn(logger, "No LFC data -> exiting early.")
  qs::qsave(tibble(), out_file)
  quit(save = "no", status = 0)
}


DEPMAP_ID <- extract_depmap_id(ccle_rna_file)
sample_info <- get_sample_info(sample_info_file, DEPMAP_ID)

lfc_genes <- unique(unlist(lfc_data$hugo_symbol))
info(logger, glue("Found {length(lfc_genes)} genes in LFC data."))

rna_data <- get_rna_expression_data(ccle_rna_file, filter_genes = lfc_genes)
mut_data <- get_mutation_data(ccle_mut_file, filter_genes = lfc_genes)
gene_cn_data <- get_gene_copy_number_data(ccle_gene_cn_file, filter_genes = lfc_genes)
segment_cn_data <- get_segment_copy_number_data(ccle_segment_cn_file, score_cn_file)


sdim <- function(x) glue("{dim(x)[[1]]}, {dim(x)[[2]]}")

info(logger, glue("Dimsions of LFC data: {sdim(lfc_data)}"))
info(logger, glue("Dimsions of RNA data: {sdim(rna_data)}"))
info(logger, glue("Dimsions of mutation data: {sdim(mut_data)}"))
info(logger, glue("Dimsions of gene CN data: {sdim(gene_cn_data)}"))
info(logger, glue("Dimsions of segment CN data: {sdim(segment_cn_data)}"))


#### ---- Logical checks of the data ---- ####

stopifnot(all(table(rna_data$hugo_symbol) == 1))
stopifnot(all(table(gene_cn_data$hugo_symbol) == 1))
stopifnot(all(table(mut_data$hugo_symbol) == 1))

stopifnot(!any(is.na(lfc_data$hugo_symbol)))
stopifnot(!any(is.na(lfc_data$replicate_id)))
stopifnot(!any(is.na(lfc_data$lfc)))


#### ---- Joining data ---- ####

join_with_rna <- function(lfc_data, rna_data) {
  info(logger, "Joining LFC data and RNA expression.")
  d <- lfc_data %>%
    left_join(rna_data, by = "hugo_symbol") %>%
    mutate(rna_expr = ifelse(is.na(rna_expr), 0, rna_expr))
  return(d)
}

join_with_mutation <- function(lfc_data, mut_data) {
  info(logger, "Joining LFC data and mutation data.")
  d <- lfc_data %>%
    left_join(mut_data, by = "hugo_symbol") %>%
    mutate(
      num_mutations = ifelse(is.na(num_mutations), 0, num_mutations)
    )
  return(d)
}

join_with_gene_copy_number <- function(lfc_data, cn_data) {
  info(logger, "Joining LFC data and gene CN data.")
  d <- lfc_data %>%
    left_join(cn_data %>% rename(copy_number = gene_cn), by = "hugo_symbol")
  return(d)
}

get_copy_number_at_chromosome_location <- function(chr, pos, segment_cn_df) {
  cn <- segment_cn_data %>%
    filter(chromosome == chr) %>%
    filter(start_pos <= !!pos) %>%
    filter(!!pos <= end_pos)

  if (nrow(cn) == 0) {
    return(NA_real_)
  } else if (nrow(cn) > 1) {
    print(cn)
    msg <- glue("Found {nrow(cn)} CN segments for chr{chr}:{pos}.")
    error(logger, msg)
    stop(msg)
  } else {
    return(cn$segment_mean[[1]])
  }
}

parse_genome_alignment <- function(df, col = genome_alignment) {
  ga <- df %>%
    pull({{ col }}) %>%
    unlist() %>%
    str_split_fixed("_", 3) %>%
    as.data.frame() %>%
    as_tibble() %>%
    select(chr = V1, pos = V2) %>%
    mutate(
      chr = str_remove(chr, "^chr"),
      pos = as.integer(pos)
    )
  bind_cols(df, ga)
}

replace_missing_cn_using_segment_cn <- function(df, segment_cn_df) {
  sgrna_cn_df <- df %>%
    distinct(sgrna, genome_alignment) %>%
    parse_genome_alignment() %>%
    select(-genome_alignment) %>%
    mutate(
      copy_number = purrr::map2_dbl(
        chr, pos,
        get_copy_number_at_chromosome_location
      )
    )

  mod_df <- df %>%
    select(-copy_number) %>%
    left_join(sgrna_cn_df, by = "sgrna")
  return(mod_df)
}

fill_in_missing_copy_number <- function(lfc_data, segment_cn) {
  info(logger, "Filling in missing copy number with segment CN data.")
  missing_cn_data <- lfc_data %>%
    filter(is.na(copy_number)) %>%
    replace_missing_cn_using_segment_cn(segment_cn)
  existing_cn_data <- lfc_data %>% filter(!is.na(copy_number))

  d <- bind_rows(missing_cn_data, existing_cn_data)
  stopifnot(nrow(d) == (nrow(missing_cn_data) + nrow(existing_cn_data)))
  return(d)
}

join_with_sample_info <- function(lfc_data, sample_info) {
  info(logger, "Joining LFC data and sample info.")
  si <- sample_info %>%
    select(lineage, primary_or_metastasis, is_male, age)
  stopifnot(nrow(si) == 1)
  return(bind_cols(lfc_data, si))
}

combined_data <- lfc_data %>%
  join_with_rna(rna_data = rna_data) %>%
  join_with_mutation(mut_data = mut_data) %>%
  join_with_gene_copy_number(cn_data = gene_cn_data) %>%
  fill_in_missing_copy_number(segment_cn = segment_cn_data) %>%
  rename(
    sgrna_target_chr = chr,
    sgrna_target_pos = pos
  ) %>%
  join_with_sample_info(sample_info = sample_info)

#### ---- Log information ---- ####

cat("\n")
br <- paste(rep("=", 80), collapse = "")
br <- paste0(br, "\n")

cat(br)
print(combined_data)
cat(br)
glimpse(combined_data)
cat(br)

info(logger, glue("Dimesions of final data: {sdim(combined_data)}"))

n_missing_rna <- sum(is.na(combined_data$rna_expr))
info(logger, glue("Number of rows missing RNA data: {n_missing_rna}."))

n_missing_cn <- sum(is.na(combined_data$copy_number))
avg_cn <- round(mean(combined_data$copy_number, na.rm = TRUE), 3)
sd_cn <- round(sd(combined_data$copy_number, na.rm = TRUE), 3)
min_cn <- round(min(combined_data$copy_number, na.rm = TRUE), 3)
max_cn <- round(max(combined_data$copy_number, na.rm = TRUE), 3)
info(logger, glue("Number of rows missing CN data: {n_missing_cn}."))
info(logger, glue("Mean (± s.d) copy number: {avg_cn} ± {sd_cn}."))
info(logger, glue("Min and max copy number: {min_cn}, {max_cn}"))

sample_info %>%
  select(
    depmap_id, sanger_model_id, cell_line_name,
    sex, age, lineage, primary_or_metastasis
  ) %>%
  knitr::kable()

#### ---- Write output ---- ####
qs::qsave(combined_data, out_file)
