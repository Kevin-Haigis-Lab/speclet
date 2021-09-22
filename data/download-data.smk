from pathlib import Path
from typing import Dict, List
import os

import requests
from requests.auth import HTTPBasicAuth
import colorama

colorama.init(autoreset=True)

data_dir = Path("data")
depmap_dir = data_dir / "depmap_21q3"
ccle_dir = data_dir / "ccle_21q3"
sanger_cosmic_dir = data_dir / "sanger-cosmic"

depmap_downloads: Dict[str, str] = {
    "Achilles_cell_line_efficacy.csv": "https://ndownloader.figshare.com/files/29124108",
    "Achilles_cell_line_growth_rate.csv": "https://ndownloader.figshare.com/files/29124111",
    "Achilles_common_essentials.csv": "https://ndownloader.figshare.com/files/29124102",
    "Achilles_dropped_guides.csv": "https://ndownloader.figshare.com/files/29124123",
    "Achilles_gene_effect.csv": "https://ndownloader.figshare.com/files/29124198",
    "Achilles_gene_effect_CERES.csv": "https://ndownloader.figshare.com/files/29124195",
    "Achilles_gene_effect_unscaled_CERES.csv": "https://ndownloader.figshare.com/files/29124486",
    "Achilles_guide_efficacy.csv": "https://ndownloader.figshare.com/files/29124483",
    "Achilles_guide_map.csv": "https://ndownloader.figshare.com/files/29124492",
    "Achilles_logfold_change.csv": "https://ndownloader.figshare.com/files/29125299",
    "Achilles_logfold_change_failures.csv": "https://ndownloader.figshare.com/files/29124513",
    "Achilles_raw_readcounts.csv": "https://ndownloader.figshare.com/files/29124612",
    "Achilles_raw_readcounts_failures.csv": "https://ndownloader.figshare.com/files/29124720",
    "Achilles_replicate_map.csv": "https://ndownloader.figshare.com/files/29124735",
    "Achilles_replicate_QC_report_failing.csv": "https://ndownloader.figshare.com/files/29124738",
    "CRISPR_common_essentials.csv": "https://ndownloader.figshare.com/files/29125284",
    "CRISPR_dataset_sources.csv": "https://ndownloader.figshare.com/files/29125290",
    "CRISPR_gene_effect.csv": "https://ndownloader.figshare.com/files/29125323",
    "CRISPR_gene_effect_CERES.csv": "https://ndownloader.figshare.com/files/29125326",
    "nonessentials.csv": "https://ndownloader.figshare.com/files/29125332",
    "common_essentials.csv": "https://ndownloader.figshare.com/files/29125281",
}

ccle_downloads: Dict[str, str] = {
    "CCLE_expression.csv": "https://ndownloader.figshare.com/files/29124747",
    "CCLE_gene_cn.csv": "https://ndownloader.figshare.com/files/29125230",
    "CCLE_segment_cn.csv": "https://ndownloader.figshare.com/files/29125278",
    "CCLE_mutations.csv": "https://ndownloader.figshare.com/files/29125233",
    "CCLE_mutations_bool_hotspot.csv": "https://ndownloader.figshare.com/files/29125245",
    "CCLE_mutations_bool_damaging.csv": "https://ndownloader.figshare.com/files/29125239",
    "CCLE_mutations_bool_nonconserving.csv": "https://ndownloader.figshare.com/files/29125248",
    "CCLE_mutations_bool_otherconserving.csv": "https://ndownloader.figshare.com/files/29125251",
    "CCLE_sample_info.csv": "https://ndownloader.figshare.com/files/29162481",
}


#### ---- Get URL for Sanger's CGC ---- ####


def get_sanger_cgc_url() -> str:
    sanger_email = os.getenv("SANGER_EMAIL")
    sanger_pass = os.getenv("SANGER_PASS")
    sanger_cgc_response = requests.get(
        "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v94/cancer_gene_census.csv",
        auth=HTTPBasicAuth(sanger_email, sanger_pass),
    )
    if sanger_cgc_response.status_code == 200:
        print(
            colorama.Fore.BLUE
            + colorama.Style.BRIGHT
            + "Successfully recieved a unique Cancer Gene Census URL."
        )
        url = sanger_cgc_response.json()["url"]
        if url is None:
            print(sanger_cgc_response.json())
            BaseException("Successul request, but no URL was returned.")
        return url
    BaseException("Unable to retrieve a unique Cancer Gene Census URL.")


CI = os.getenv("CI")

if CI is None:
    sanger_cgc_url = get_sanger_cgc_url()
else:
    print(colorama.Fore.YELLOW + f"CI: using fake URL for Sanger CGC file.")
    sanger_cgc_url = "www.mock-url.com"


#### ---- Rules ---- ####


rule all:
    input:
        depmap_files=expand(
            depmap_dir / "{filename}", filename=list(depmap_downloads.keys())
        ),
        ccle_files=expand(ccle_dir / "{filename}", filename=list(ccle_downloads.keys())),
        sanger_cgc=sanger_cosmic_dir / "cancer_gene_census.csv",


rule download_depmap:
    output:
        filename=depmap_dir / "{filename}",
    params:
        url=lambda w: depmap_downloads[w.filename],
    shell:
        "wget --output-document {output.filename} {params.url}"


rule download_ccle:
    output:
        filename=ccle_dir / "{filename}",
    params:
        url=lambda w: ccle_downloads[w.filename],
    shell:
        "wget --output-document {output.filename} {params.url}"


rule download_sanger:
    output:
        cgc_filename=sanger_cosmic_dir / "cancer_gene_census.tsv",
    params:
        url=sanger_cgc_url,
    shell:
        "wget --output-document {output.cgc_filename} {params.url}"
