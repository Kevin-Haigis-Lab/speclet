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

# Simple basic preparation steps for LFC data frame.
prepare_lfc_data <- function(lfc_df, screen_scoure) {
  lfc_df %>%
    add_column(screen = screen_scoure) %>%
    filter(!is.na(hugo_symbol)) %>%
    distinct()
}

# Retrieve the LFC data from Achilles and Sanger.
# If data is available from both sources, they are merged. If
# neither data source is available, `NULL` is returned.
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

# Filter out sgRNA with multiple targets.
# Returns early if the input `lfc_df` is `NULL`.
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

# Parse the genome alignment for a sgRNA into
# chromosome `chr` and position `pos`. Returns
# early if the input `lfc_df` is `NULL`.
parse_genome_alignment <- function(df, col = genome_alignment) {
  info(logger, "Parsing genome alignment of sgRNAs.")
  if (is.null(df)) {
    return(df)
  }

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

# Pull the DepMap ID from the RNA expression file name.
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

# Get the sample information for a cell line.
# If no sample information is available, then `NULL`
# is returned. This can happen if the cell line
# was removed in an earlier munging step because it
# was "engineered" or non-cancerous in origin.
get_sample_info <- function(sample_info_path, depmap_id) {
  info(logger, "Retrieving sample information.")
  sample_info <- read_csv(sample_info_path) %>%
    filter(depmap_id == !!depmap_id)

  if (nrow(sample_info) == 0) {
    msg <- glue("Did not find sample information for DepMap ID '{depmap_id}'.")
    warn(logger, msg)
    return(NULL)
  } else if (nrow(sample_info) > 1) {
    print(sample_info)
    msg <- glue("Found {nrow(sample_info)} samples with DepMap ID '{depmap_id}'.")
    error(logger, msg)
    stop(msg)
  }

  return(sample_info)
}

# Retrieve RNA expression data.
# If a list of genes is supplied in `filter_genes`, then
# all other genes are removed from the data frame.
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

# Retrieve and summarize mutation data.
# The data is adjusted so that each gene is only present
# in a single row. If it has multiple mutations, these
# are summarised. If a list of genes is supplied in
# `filter_genes`, then all other genes are removed from
# the data frame.
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

# Retrive gene copy number data from the CCLE.
# If a list of genes is supplied in `filter_genes`, then
# all other genes are removed from the data frame.
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

# Retrieve segment copy number data from CCLE and Sanger.
# If data from both sources is available, they are merged.
# If neither source provided segment CN data, `NULL` is
# returned.
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
    warn(logger, msg)
    return(NULL)
  } else if (nrow(ccle_cn) == 0) {
    info(logger, "Segment CN found from Sanger.")
    return(ccle_cn)
  } else if (nrow(ccle_cn) == 0) {
    info(logger, "Segment CN found from CCLE.")
    return(ccle_cn)
  } else {
    info(logger, "Data found from both CCLE and Sanger.")
    cn <- bind_rows(ccle_cn, sanger_cn) %>% distinct()
    return(cn)
  }
}

# Helper function to quit early (usually because no
# LFC data or sample info).
quit_early <- function(reason, out_file) {
  warn(logger, reason)
  qs::qsave(tibble::tibble(), out_file)
  quit(save = "no", status = 0)
}


lfc_data <- get_log_fold_change_data(achilles_lfc_file, score_lfc_file) %>%
  remove_sgrna_that_target_multiple_genes() %>%
  parse_genome_alignment()

if (is.null(lfc_data)) {
  quit_early(
    reason = "No LFC data - exiting early.",
    out_file = out_file
  )
}

DEPMAP_ID <- extract_depmap_id(ccle_rna_file)
sample_info <- get_sample_info(sample_info_file, DEPMAP_ID)

if (is.null(sample_info)) {
  quit_early(
    reason = "No sample info - exiting early.",
    out_file = out_file
  )
}

lfc_genes <- unique(unlist(lfc_data$hugo_symbol))
info(logger, glue("Found {length(lfc_genes)} genes in LFC data."))

rna_data <- get_rna_expression_data(ccle_rna_file, filter_genes = lfc_genes)
mut_data <- get_mutation_data(ccle_mut_file, filter_genes = lfc_genes)
gene_cn_data <- get_gene_copy_number_data(ccle_gene_cn_file, filter_genes = lfc_genes)
segment_cn_data <- get_segment_copy_number_data(ccle_segment_cn_file, score_cn_file)


sdim <- function(x) glue("{dim(x)[[1]]}, {dim(x)[[2]]}")

info(logger, glue("Dimensions of LFC data: {sdim(lfc_data)}"))
info(logger, glue("Dimensions of RNA data: {sdim(rna_data)}"))
info(logger, glue("Dimensions of mutation data: {sdim(mut_data)}"))
info(logger, glue("Dimensions of gene CN data: {sdim(gene_cn_data)}"))
if (!is.null(segment_cn_data)) {
  info(logger, glue("Dimensions of segment CN data: {sdim(segment_cn_data)}"))
} else {
  info(logger, glue("No segment CN data."))
}


#### ---- Logical checks of the data ---- ####

stopifnot(all(table(rna_data$hugo_symbol) == 1))
stopifnot(all(table(gene_cn_data$hugo_symbol) == 1))
stopifnot(all(table(mut_data$hugo_symbol) == 1))

stopifnot(!any(is.na(lfc_data$hugo_symbol)))
stopifnot(!any(is.na(lfc_data$replicate_id)))
stopifnot(!any(is.na(lfc_data$lfc)))


#### ---- Joining data ---- ####

# Merge LFC data with RNA expression data.
# Genes with no RNA expression are assumed to be unexpressed.
join_with_rna <- function(lfc_data, rna_data) {
  info(logger, "Joining LFC data and RNA expression.")
  d <- lfc_data %>%
    left_join(rna_data, by = "hugo_symbol") %>%
    mutate(rna_expr = ifelse(is.na(rna_expr), 0, rna_expr))
  return(d)
}

# Merge LFC data with mutation data.
# The `num_mutations` and `is_mutated` columns are filled with
# `0` and `FALSE`, respectively, but other columns are
# left as `NA`.
join_with_mutation <- function(lfc_data, mut_data) {
  info(logger, "Joining LFC data and mutation data.")
  d <- lfc_data %>%
    left_join(mut_data, by = "hugo_symbol") %>%
    mutate(
      num_mutations = ifelse(is.na(num_mutations), 0, num_mutations),
      is_mutated = ifelse(is.na(is_mutated), FALSE, is_mutated)
    )
  return(d)
}

# Merge LFC data with gene copy number.
join_with_gene_copy_number <- function(lfc_data, cn_data) {
  info(logger, "Joining LFC data and gene CN data.")
  d <- lfc_data %>%
    left_join(cn_data %>% rename(copy_number = gene_cn), by = "hugo_symbol")
  return(d)
}

# Reconcile cases with multiple segment mean values.
# Below are the rules used:
#  1. Remove any missing data and see if that reduces the data to a single point.
#  2. If there are only two values:
#   a. If the difference is small, return the mean.
#   b. Else, return the CN with the most number of probes (taking the mean if there are still multiple values).
#  3. If there are more than 2 values, return the median CN.
reconcile_multiple_segment_copy_numbers <- function(cn_df) {
  info(logger, "Reconciling multiple CN data for a single position.")

  cn_df <- cn_df %>% dplyr::filter(!is.na(segment_mean))
  if (nrow(cn_df) == 1) {
    return(cn_df$segment_mean[[1]])
  }

  cn_vals <- unlist(cn_df$segment_mean)
  log4r::debug(logger, glue("Found {length(cn_vals)} copy number values."))

  if (lenth(cn_vals) == 2) {
    if (cn_vals[[1]] - cn_vals[[2]] < 0.25) {
      log4r::debug(logger, "Returning mean of two values.")
      return(mean(cn_vals))
    } else {
      log4r::debug(logger, "Returning mean of CN values with most probes.")
      cn <- cn_df %>%
        filter(num_probes == max(num_probes)) %>%
        pull(segment_mean) %>%
        unlist() %>%
        mean()
      return(unlist(cn))
    }
  } else {
    log4r::debug(logger, "Returning mean of >2 CN values.")
    return(median(cn_vals))
  }
}

# Get the copy number at a postion `pos` on a chromosome `chr`.
# If the position is not included in `segment_cn_df`, then the
# value of `NA` is returned.
# Will throw an error if multiple CN values are found so
# that that instance can be addressed if need be.
get_copy_number_at_chromosome_location <- function(chr, pos, segment_cn_df) {
  cn <- segment_cn_df %>%
    dplyr::filter(chromosome == chr) %>%
    dplyr::filter(start_pos <= !!pos) %>%
    dplyr::filter(!!pos <= end_pos)

  if (nrow(cn) == 0) {
    return(NA_real_)
  } else if (nrow(cn) > 1) {
    print(cn)
    warn(logger, glue("Found {nrow(cn)} CN segments for chr{chr}:{pos}."))
    cn_val <- reconcile_multiple_segment_copy_numbers(cn)
    warn(logger, glue("Reconciled multiple values to {round(cn_val, 3)}."))
  } else {
    return(cn$segment_mean[[1]])
  }
}

# Replace missing CN data using segment CN data.
# Requires that the data frame `df` have columns
# for chromosome `chr` and position `pos`.
replace_missing_cn_using_segment_cn <- function(df, segment_cn_df) {
  sgrna_cn_df <- df %>%
    distinct(sgrna, chr, pos) %>%
    mutate(copy_number = purrr::map2_dbl(
      chr,
      pos,
      get_copy_number_at_chromosome_location,
      segment_cn_df = segment_cn_df
    )) %>%
    select(sgrna, copy_number)

  mod_df <- df %>%
    select(-copy_number) %>%
    left_join(sgrna_cn_df, by = "sgrna")
  return(mod_df)
}

# Fill in the missing CN data in the LFC data frame.
# Rows with missing CN are passed to `replace_missing_cn_using_segment_cn()`
# and those with CN data are left as is.
fill_in_missing_copy_number <- function(lfc_data, segment_cn) {
  info(logger, "Filling in missing copy number with segment CN data.")

  if (is.null(segment_cn)) {
    warn(logger, "No segment CN data - returning early.")
    return(lfc_data)
  }

  missing_cn_data <- lfc_data %>%
    filter(is.na(copy_number)) %>%
    replace_missing_cn_using_segment_cn(segment_cn)
  existing_cn_data <- lfc_data %>% filter(!is.na(copy_number))

  d <- bind_rows(missing_cn_data, existing_cn_data)
  stopifnot(nrow(d) == (nrow(missing_cn_data) + nrow(existing_cn_data)))
  return(d)
}

# Merge LFC data with sample information.
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
