import os
from pathlib import Path

import pandas as pd

temp_dir = Path("/n/scratch3/users/j/jc604/speclet/modeling-data-processing")
save_dir = Path("modeling_data")
raw_input_dir = Path("data/depmap_20q3")
input_data_dir = Path("modeling_data")

# Get all DepMapIDs from the logFC data.
# run `munge/list_all_depmapids.R` to update the file if data is updated.

all_depmapids = pd.read_csv(input_data_dir / "all_achilles_depmapids.csv").depmap_id.to_list()


rule all:
    input:
        save_dir / "depmap_modeling_dataframe.csv"

rule prepare_depmap_data:
    input:
        sample_info = raw_input_dir / "sample_info.csv",
        guide_map = raw_input_dir / "Achilles_guide_map.csv",
        dropped_guides = raw_input_dir / "Achilles_dropped_guides.csv",
        replicate_map = raw_input_dir / "Achilles_replicate_map.csv",
        logfold_change = raw_input_dir / "Achilles_logfold_change.csv",
        ccle_mutations = raw_input_dir / "CCLE_mutations.csv",
        ccle_segment_cn = raw_input_dir / "CCLE_segment_cn.csv",
        ccle_gene_cn = raw_input_dir / "CCLE_gene_cn.csv",
        ccle_fusions = raw_input_dir / "CCLE_fusions.csv",
        ccle_expression = raw_input_dir / "CCLE_expression_full_v2.csv",
        gene_effect = raw_input_dir / "Achilles_gene_effect.csv",
        gene_effect_unscaled = raw_input_dir / "Achilles_gene_effect_unscaled.csv",
    output:
        sample_info = save_dir / "sample_info.csv",
        guide_map = save_dir / "achilles_guide_map.csv",
        replicate_map = save_dir / "achilles_replicate_map.csv",
        logfold_change = save_dir / "achilles_logfold_change.csv",
        ccle_mutations = save_dir / "ccle_mutations.csv",
        kras_mutations = save_dir / "kras_mutations.csv",
        ccle_semgent_cn = save_dir / "ccle_semgent_cn.csv",
        ccle_gene_cn = save_dir / "ccle_gene_cn.csv",
        ccle_fusions = save_dir / "ccle_fusions.csv",
        ccle_expression = save_dir / "ccle_expression.csv",
        achilles_gene_effect = save_dir / "achilles_gene_effect.csv",
        notebook_md = "munge/005_prepare-depmap-data.md"
    shell:
        "jupyter nbconvert --to notebook --inplace --execute munge/005_prepare-depmap-data.ipynb && "
        "nbqa black munge/005_prepare-depmap-data.ipynb --nbqa-mutate && "
        "nbqa isort munge/005_prepare-depmap-data.ipynb --nbqa-mutate && "
        "jupyter nbconvert --to markdown munge/005_prepare-depmap-data.ipynb"

# Prepare sample information.
rule prep_sampleinfo:
    input:
        ccle_mutations = input_data_dir / "ccle_mutations.csv",
        kras_mutations = input_data_dir / "kras_mutations.csv",
        sample_info = input_data_dir / "sample_info.csv",
    output:
        out_file = temp_dir / "achilles-sample-info.qs"
    script:
        "014_make-achilles-sample-info.R"


# Split segment mean CN data by DepMapID.
rule split_segmentcn:
    input:
        data_file = input_data_dir / "ccle_semgent_cn.csv"
    output:
        out_files = expand(
            (temp_dir / "segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    script:
        "015_split-modeling-data.R"


# Split CNA data by DepMapID.
rule split_genecn:
    input:
        data_file = input_data_dir / "ccle_gene_cn.csv"
    output:
        out_files = expand(
            (temp_dir / "genecn_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    script:
        "015_split-modeling-data.R"


# Split mutation data by DepMapID.
rule split_mut:
    input:
        data_file = input_data_dir / "ccle_mutations.csv"
    output:
        out_files = expand(
            (temp_dir / "mut_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    script:
        "015_split-modeling-data.R"


# Split logFC data by DepMapID.
rule split_logfc:
    input:
        data_file = input_data_dir / "achilles_logfold_change.csv"
    output:
        out_files = expand(
            (temp_dir / "logfc_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    script:
        "015_split-modeling-data.R"


# Split RNA data by DepMapID.
rule split_rna:
    input:
        data_file = input_data_dir / "ccle_expression.csv"
    output:
        out_files = expand(
            (temp_dir / "rna_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    script:
        "015_split-modeling-data.R"



# Merge all data for a DepMapID.
rule merge_data:
    input:
        segmentcn_file = temp_dir / "segmentcn_{depmapid}.qs",
        genecn_file = temp_dir / "genecn_{depmapid}.qs",
        mut_file = temp_dir / "mut_{depmapid}.qs",
        logfc_file = temp_dir / "logfc_{depmapid}.qs",
        rna_file = temp_dir / "rna_{depmapid}.qs",
        sampleinfo_file = temp_dir / "achilles-sample-info.qs"
    output:
        out_file = temp_dir / "merged_{depmapid}.qs"
    script:
        "016_merge-modeling-data.R"



# Combine the data for a single DepMapID.
rule combine_data:
    input:
        input_files = expand(
            (temp_dir / "merged_{depmapid}.qs").as_posix(),
            depmapid=all_depmapids
        )
    output:
        out_file = save_dir / "depmap_modeling_dataframe.csv"
    script:
        "017_combine-modeling-data.R"

rule auxillary_files:
    input:
        guide_efficacy = raw_input_dir / "Achilles_guide_efficacy.csv",
        common_essentials = raw_input_dir / "common_essentials.csv",
        nonessentials = raw_input_dir / "nonessentials.csv",

    output:
        guide_efficacy = save_dir / "achilles_guide_efficacy.csv",
        achilles_essential_genes = save_dir / "achilles_essential_genes.csv"
    script:
        "021_prepare-depmap-auxillary-files.R"

rule isolate_crc:
    input:
        depmap_modeling_data = save_dir / "depmap_modeling_dataframe.csv"
    output:
        depmap_CRC_data = save_dir / "depmap_CRC_data.csv",
        depmap_CRC_data_subsample = save_dir / "depmap_CRC_data_subsample.csv",
        depmap_CRC_data_largesubsample = save_dir / "depmap_CRC_data_largesubsample.csv"

    script:
        "025_isolate-crc-data.R"

rule make_test_data:
    input:
        depmap_modeling_df = save_dir / "depmap_modeling_dataframe.csv"
    output:
        output_dest = Path("tests", "depmap_test_data.csv")
    script:
        "030_test-data.R"
