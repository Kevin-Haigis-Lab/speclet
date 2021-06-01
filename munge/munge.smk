# A Snakemake pipeline for preparing raw data for analysis.

from pathlib import Path

DATA_DIR = Path("data")
DEPMAP_DIR = DATA_DIR / "depmap_21q2"
SCORE_DIR = DATA_DIR / "score_21q2"
CCLE_DIR = DATA_DIR / "ccle_21q2"

MODELING_DATA_DIR = Path("modeling_data")

rule all:
    input:
        # tidy_ccle
        rna_expr = MODELING_DATA_DIR / "ccle_expression.csv",
        gene_cn = MODELING_DATA_DIR / "ccle_gene_cn.csv",
        gene_mutations = MODELING_DATA_DIR / "ccle_mutations.csv",
        sample_info = MODELING_DATA_DIR / "ccle_sample_info.csv",
        known_essentials = MODELING_DATA_DIR / "known_essentials.csv",
        # tidy_depmap
        achilles_log_fold_change = MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv",
        achilles_gene_effect = MODELING_DATA_DIR / "achilles_gene_effect.csv",
        chronos_gene_effect = MODELING_DATA_DIR / "chronos_gene_effect.csv",
        # tidy score
        copy_number = MODELING_DATA_DIR / "score_gene_cn.csv",
        gene_effect = MODELING_DATA_DIR / "score_gene_effect.csv",
        log_fold_change = MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",


rule tidy_ccle:
    input:
        rna_expr = CCLE_DIR / "CCLE_expression.csv",
        gene_cn = CCLE_DIR / "CCLE_gene_cn.csv",
        gene_mutations = CCLE_DIR / "CCLE_mutations.csv",
        sample_info = CCLE_DIR / "CCLE_sample_info.csv",
    output:
        rna_expr = MODELING_DATA_DIR / "ccle_expression.csv",
        gene_cn = MODELING_DATA_DIR / "ccle_gene_cn.csv",
        gene_mutations = MODELING_DATA_DIR / "ccle_mutations.csv",
        sample_info = MODELING_DATA_DIR / "ccle_sample_info.csv",
    script:
        "005_prepare-ccle-raw-data.R"

rule tidy_depmap:
    input:
        common_essentials = DEPMAP_DIR / "common_essentials.csv",
        nonessentials = DEPMAP_DIR / "nonessentials.csv",
        achilles_dropped_guides = DEPMAP_DIR / "Achilles_dropped_guides.csv",
        achilles_guide_efficacy = DEPMAP_DIR / "Achilles_guide_efficacy.csv",
        achilles_guide_map = DEPMAP_DIR / "Achilles_guide_map.csv",
        achilles_gene_effect = DEPMAP_DIR / "Achilles_gene_effect.csv",
        achilles_gene_effect_unscaled = DEPMAP_DIR / "Achilles_gene_effect_unscaled.csv",
        achilles_logfold_change = DEPMAP_DIR / "Achilles_logfold_change.csv",
        achilles_replicate_map = DEPMAP_DIR / "Achilles_replicate_map.csv",
        all_gene_effect_chronos = DEPMAP_DIR / "CRISPR_gene_effect_Chronos.csv",
    output:
        known_essentials = MODELING_DATA_DIR / "known_essentials.csv",
        achilles_log_fold_change = MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv",
        achilles_gene_effect = MODELING_DATA_DIR / "achilles_gene_effect.csv",
        chronos_gene_effect = MODELING_DATA_DIR / "chronos_gene_effect.csv",
    script:
        "007_prepare-dempap-raw-data.R"

rule tidy_score:
    input:
        copy_number = SCORE_DIR / "SCORE_copy_number.csv",
        gene_effect = SCORE_DIR / "SCORE_gene_effect.csv",
        gene_effect_unscaled = SCORE_DIR / "SCORE_gene_effect_unscaled.csv",
        log_fold_change = SCORE_DIR / "SCORE_logfold_change.csv",
        guide_map = SCORE_DIR / "SCORE_guide_gene_map.csv",
        replicate_map = SCORE_DIR / "SCORE_replicate_map.csv",
    output:
        copy_number = MODELING_DATA_DIR / "score_gene_cn.csv",
        gene_effect = MODELING_DATA_DIR / "score_gene_effect.csv",
        log_fold_change = MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
    script:
        "009_prepare-score-raw-data.R"
