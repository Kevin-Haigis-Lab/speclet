# A Snakemake pipeline for preparing raw data for analysis.

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DEPMAP_DIR = DATA_DIR / "depmap_21q2"
SCORE_DIR = DATA_DIR / "score_21q2"
CCLE_DIR = DATA_DIR / "ccle_21q2"

MODELING_DATA_DIR = Path("modeling_data")

TEMP_DIR = Path("/n/scratch3/users/j/jc604/speclet/munge-intermediates")
# TEMP_DIR = Path("temp")

ENVIRONMENT_YAML = "pipeline-environment.yml"

all_depmap_ids = pd.read_csv(DATA_DIR / "all-depmap-ids.csv").depmap_id.to_list()
# all_depmap_ids = all_depmap_ids[:10] ### TESTING ###
# all_depmap_ids += ["ACH-002227", "ACH-001738"]

rule all:
    input:
        # Prep CCLE data
        rna_expr = MODELING_DATA_DIR / "ccle_expression.csv",
        segment_cn = MODELING_DATA_DIR / "ccle_segment_cn.csv",
        gene_cn = MODELING_DATA_DIR / "ccle_gene_cn.csv",
        gene_mutations = MODELING_DATA_DIR / "ccle_mutations.csv",
        sample_info = MODELING_DATA_DIR / "ccle_sample_info.csv",
        known_essentials = MODELING_DATA_DIR / "known_essentials.csv",
        # Prep DepMap data
        achilles_log_fold_change = MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv",
        achilles_gene_effect = MODELING_DATA_DIR / "achilles_gene_effect.csv",
        chronos_gene_effect = MODELING_DATA_DIR / "chronos_gene_effect.csv",
        # Prep SCORE data.
        copy_number = MODELING_DATA_DIR / "score_segment_cn.csv",
        gene_effect = MODELING_DATA_DIR / "score_gene_effect.csv",
        log_fold_change = MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
        # Modeling data.
        full_modeling_dataframe = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv",
        check_notebook_md = "munge/017_check-depmap-modeling-data.md",



rule tidy_ccle:
    input:
        rna_expr = CCLE_DIR / "CCLE_expression.csv",
        segment_cn = CCLE_DIR / "CCLE_segment_cn.csv",
        gene_cn = CCLE_DIR / "CCLE_gene_cn.csv",
        gene_mutations = CCLE_DIR / "CCLE_mutations.csv",
        sample_info = CCLE_DIR / "CCLE_sample_info.csv",
    output:
        rna_expr = MODELING_DATA_DIR / "ccle_expression.csv",
        segment_cn = MODELING_DATA_DIR / "ccle_segment_cn.csv",
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
        copy_number = MODELING_DATA_DIR / "score_segment_cn.csv",
        gene_effect = MODELING_DATA_DIR / "score_gene_effect.csv",
        log_fold_change = MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
    script:
        "009_prepare-score-raw-data.R"


rule split_ccle_rna_expression:
    input:
        data_file = MODELING_DATA_DIR / "ccle_expression.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "ccle-rna_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_gene_cn:
    input:
        data_file = MODELING_DATA_DIR / "ccle_gene_cn.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "ccle-genecn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_segment_cn:
    input:
        data_file = MODELING_DATA_DIR / "ccle_segment_cn.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "ccle-segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_mutations:
    input:
        data_file = MODELING_DATA_DIR / "ccle_mutations.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "ccle-mut_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


rule split_achilles_lfc:
    input:
        data_file = MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "achilles-lfc_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


rule split_score_cn:
    input:
        data_file = MODELING_DATA_DIR / "score_segment_cn.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "score-segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"

rule split_score_lfc:
    input:
        data_file = MODELING_DATA_DIR / "score_log_fold_change_filtered.csv"
    output:
        out_files = expand(
            (TEMP_DIR / "score-lfc_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    script:
        "011_split-file-by-depmapid.R"


# Merge all data for a DepMapID.
rule merge_data:
    input:
        ccle_rna = TEMP_DIR / "ccle-rna_{depmapid}.qs",
        ccle_gene_cn = TEMP_DIR / "ccle-genecn_{depmapid}.qs",
        ccle_segment_cn = TEMP_DIR / "ccle-segmentcn_{depmapid}.qs",
        ccle_mut = TEMP_DIR / "ccle-mut_{depmapid}.qs",
        achilles_lfc = TEMP_DIR / "achilles-lfc_{depmapid}.qs",
        score_cn = TEMP_DIR / "score-segmentcn_{depmapid}.qs",
        score_lfc = TEMP_DIR / "score-lfc_{depmapid}.qs",
        sample_info = MODELING_DATA_DIR / "ccle_sample_info.csv",
    output:
        out_file = TEMP_DIR / "merged_{depmapid}.qs"
    script:
        "013_merge-modeling-data.R"


rule combine_data:
    input:
        input_files = expand(
            (TEMP_DIR / "merged_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids
        )
    output:
        out_file = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv"
    script:
        "015_combine-modeling-data.R"

rule check_depmap_modeling_data:
    input:
        modeling_df = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv",
        check_nb = "munge/017_check-depmap-modeling-data.ipynb",
    output:
        output_md = "munge/017_check-depmap-modeling-data.md"
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.check_nb} && "
        "nbqa black {input.check_nb} --nbqa-mutate && "
        "nbqa isort {input.check_nb} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.check_nb}"
