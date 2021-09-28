# Re-usable code for munging data.

# ---- General data processing ----

extract_hugo_gene_name <- function(df, col = gene) {
  df %>%
    mutate({{ col }} := str_remove({{ col }}, " \\(.*"))
}


flatten_wide_df_by_gene <- function(df,
                                    values_to,
                                    id_col_name = "X1",
                                    rename_id_col_to = depmap_id,
                                    col_names_to = hugo_symbol) {
  df %>%
    dplyr::rename({{ rename_id_col_to }} := !!id_col_name) %>%
    pivot_longer(
      -{{ rename_id_col_to }},
      names_to = "names__col",
      values_to = values_to
    ) %>%
    extract_hugo_gene_name(names__col) %>%
    rename({{ col_names_to }} := names__col)
}


remove_columns <- function(df, cols_to_drop) {
  return(df[, !(colnames(df) %in% cols_to_drop)])
}


# ---- Reading Achilles data ----

read_achilles_replicate_map <- function(f) {
  readr::read_csv(f) %>%
    janitor::clean_names() %>%
    rename(depmap_id = dep_map_id)
}


get_dropped_replicates <- function(rep_map) {
  rep_map %>%
    filter(!passes_qc) %>%
    pull(replicate_id) %>%
    unlist() %>%
    unique()
}


read_achilles_guide_map <- function(f) {
  readr::read_csv(f) %>%
    janitor::clean_names() %>%
    extract_hugo_gene_name() %>%
    rename(hugo_symbol = gene)
}


# ----Reading SCORE data ----


read_score_replicate_map <- function(path) {
  readr::read_csv(path) %>%
    janitor::clean_names() %>%
    dplyr::rename(depmap_id = dep_map_id)
}

read_score_guide_map <- function(f) {
  readr::read_csv(f) %>%
    extract_hugo_gene_name(gene) %>%
    rename(hugo_symbol = gene)
}

get_score_read_count_path <- function(dir, replicate_id) {
  file.path(dir, paste0(replicate_id, ".read_count.tsv.gz"))
}

read_score_count_file <- function(path) {
  readr::read_tsv(path) %>%
    rename(sgrna_id = sgRNA, hugo_symbol = gene)
}
