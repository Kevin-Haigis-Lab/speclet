import os
import pandas as pd
from pathlib import Path

temp_dir = Path("/n/scratch3/users/j/jc604/speclet/modeling-data-processing")
save_dir = Path('modeling_data')
input_data_dir = Path("modeling_data")

# Get all DepMapIDs from the logFC data.
# run `munge/list_all_depmapids.R` to update the file if data is updated.
all_depmapids = pd.read_csv(input_data_dir / "all_achilles_depmapids.csv").depmap_id.to_list()


rule all:
    input:
        (save_dir / "depmap_modeling_dataframe.csv").as_posix()


# Prepare sample information.
rule prep_sampleinfo:
    output:
        out_file = (temp_dir / "achilles-sample-info.qs").as_posix()
    script:
        "014_make-achilles-sample-info.R"


# Split CNA data by DepMapID.
rule split_cna:
    input:
        data_file = (input_data_dir / "ccle_semgent_cn.csv").as_posix()
    output:
        out_files = expand((temp_dir / "cna_{depmapid}.qs").as_posix(), depmapid=all_depmapids)
    script:
        "015_split-modeling-data.R"


# Split mutation data by DepMapID.
rule split_mut:
    input:
        data_file = (input_data_dir / "ccle_mutations.csv").as_posix()
    output:
        out_files = expand((temp_dir / "mut_{depmapid}.qs").as_posix(), depmapid=all_depmapids)
    script:
        "015_split-modeling-data.R"


# Split logFC data by DepMapID.
rule split_logfc:
    input:
        data_file = (input_data_dir / "achilles_logfold_change.csv").as_posix()
    output:
        out_files = expand((temp_dir / "logfc_{depmapid}.qs").as_posix(), depmapid=all_depmapids)
    script:
        "015_split-modeling-data.R"


# Split RNA data by DepMapID.
rule split_rna:
    input:
        data_file = (input_data_dir / "ccle_expression.csv").as_posix()
    output:
        out_files = expand((temp_dir / "rna_{depmapid}.qs").as_posix(), depmapid=all_depmapids)
    script:
        "015_split-modeling-data.R"



# Merge all data for a DepMapID.
rule merge_data:
    input:
        cna_file = (temp_dir / "cna_{depmapid}.qs").as_posix(),
        mut_file = (temp_dir / "mut_{depmapid}.qs").as_posix(),
        logfc_file = (temp_dir / "logfc_{depmapid}.qs").as_posix(),
        rna_file = (temp_dir / "rna_{depmapid}.qs").as_posix(),
        sampleinfo_file = (temp_dir / "achilles-sample-info.qs").as_posix()
    output:
        out_file = (temp_dir / "merged_{depmapid}.qs").as_posix()
    script:
        "016_merge-modeling-data.R"



# Combine the data for a single DepMapID.
rule combine_data:
    input:
        input_files = expand((temp_dir / "merged_{depmapid}.qs").as_posix(), depmapid=all_depmapids)
    output:
        out_file = (save_dir / "depmap_modeling_dataframe.csv").as_posix()
    script:
        "017_combine-modeling-data.R"
