
extract_hugo_gene_name <- function(df, col = gene) {
  df %>%
    mutate({{ col }} := str_remove({{ col }}, " \\(.*"))
}


flatten_wide_df_by_gene <- function(df, values_to) {
  df %>%
    rename(depmap_id = X1) %>%
    pivot_longer(-depmap_id, names_to = "hugo_symbol", values_to = values_to) %>%
    extract_hugo_gene_name(hugo_symbol)
}


remove_columns <- function(df, cols_to_drop) {
  return(df[, !(colnames(df) %in% cols_to_drop)])
}


read_replicate_map <- function(f) {
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


read_guide_map <- function(f) {
  readr::read_csv(f) %>%
    janitor::clean_names() %>%
    extract_hugo_gene_name() %>%
    rename(hugo_symbol = gene)
}
