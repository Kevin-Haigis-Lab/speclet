
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
