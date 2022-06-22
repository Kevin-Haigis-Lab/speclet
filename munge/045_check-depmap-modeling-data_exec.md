# Check of modeling data


```python
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, progress
```


```python
client = Client(n_workers=4, threads_per_worker=2, memory_limit="32GB")
client
```

    2022-06-22 14:19:38,777 - distributed.diskutils - INFO - Found stale lock file and directory '/n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-bvxablb0', purging
    2022-06-22 14:19:38,806 - distributed.diskutils - INFO - Found stale lock file and directory '/n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-tlv8wp0b', purging





<div>
    <div style="width: 24px; height: 24px; background-color: #e1e1e1; border: 3px solid #9D9D9D; border-radius: 5px; position: absolute;"> </div>
    <div style="margin-left: 48px;">
        <h3 style="margin-bottom: 0px;">Client</h3>
        <p style="color: #9D9D9D; margin-bottom: 0px;">Client-dfe4d0ce-f257-11ec-814d-149ecf16877d</p>
        <table style="width: 100%; text-align: left;">

        <tr>

            <td style="text-align: left;"><strong>Connection method:</strong> Cluster object</td>
            <td style="text-align: left;"><strong>Cluster type:</strong> distributed.LocalCluster</td>

        </tr>


            <tr>
                <td style="text-align: left;">
                    <strong>Dashboard: </strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                </td>
                <td style="text-align: left;"></td>
            </tr>


        </table>


            <details>
            <summary style="margin-bottom: 20px;"><h3 style="display: inline;">Cluster Info</h3></summary>
            <div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-mod-trusted jp-OutputArea-output">
    <div style="width: 24px; height: 24px; background-color: #e1e1e1; border: 3px solid #9D9D9D; border-radius: 5px; position: absolute;">
    </div>
    <div style="margin-left: 48px;">
        <h3 style="margin-bottom: 0px; margin-top: 0px;">LocalCluster</h3>
        <p style="color: #9D9D9D; margin-bottom: 0px;">6467eec9</p>
        <table style="width: 100%; text-align: left;">
            <tr>
                <td style="text-align: left;">
                    <strong>Dashboard:</strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                </td>
                <td style="text-align: left;">
                    <strong>Workers:</strong> 4
                </td>
            </tr>
            <tr>
                <td style="text-align: left;">
                    <strong>Total threads:</strong> 8
                </td>
                <td style="text-align: left;">
                    <strong>Total memory:</strong> 119.21 GiB
                </td>
            </tr>

            <tr>
    <td style="text-align: left;"><strong>Status:</strong> running</td>
    <td style="text-align: left;"><strong>Using processes:</strong> True</td>
</tr>


        </table>

        <details>
            <summary style="margin-bottom: 20px;">
                <h3 style="display: inline;">Scheduler Info</h3>
            </summary>

            <div style="">
    <div>
        <div style="width: 24px; height: 24px; background-color: #FFF7E5; border: 3px solid #FF6132; border-radius: 5px; position: absolute;"> </div>
        <div style="margin-left: 48px;">
            <h3 style="margin-bottom: 0px;">Scheduler</h3>
            <p style="color: #9D9D9D; margin-bottom: 0px;">Scheduler-e7379a1d-1481-40b2-a909-d3b6be83fd8c</p>
            <table style="width: 100%; text-align: left;">
                <tr>
                    <td style="text-align: left;">
                        <strong>Comm:</strong> tcp://127.0.0.1:33892
                    </td>
                    <td style="text-align: left;">
                        <strong>Workers:</strong> 4
                    </td>
                </tr>
                <tr>
                    <td style="text-align: left;">
                        <strong>Dashboard:</strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                    </td>
                    <td style="text-align: left;">
                        <strong>Total threads:</strong> 8
                    </td>
                </tr>
                <tr>
                    <td style="text-align: left;">
                        <strong>Started:</strong> Just now
                    </td>
                    <td style="text-align: left;">
                        <strong>Total memory:</strong> 119.21 GiB
                    </td>
                </tr>
            </table>
        </div>
    </div>

    <details style="margin-left: 48px;">
        <summary style="margin-bottom: 20px;">
            <h3 style="display: inline;">Workers</h3>
        </summary>


        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 0</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:42837
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 2
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:43415/status" target="_blank">http://127.0.0.1:43415/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 29.80 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:33033
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-91bw34nx
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 1</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:46733
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 2
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:45956/status" target="_blank">http://127.0.0.1:45956/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 29.80 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:41014
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-f1t_hati
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 2</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:36788
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 2
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:35840/status" target="_blank">http://127.0.0.1:35840/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 29.80 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:38395
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-o986ch3t
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 3</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:34655
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 2
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:38084/status" target="_blank">http://127.0.0.1:38084/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 29.80 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:39735
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /n/data1/hms/dbmi/park/Cook/speclet/munge/dask-worker-space/worker-ams7d6y9
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>


    </details>
</div>

        </details>
    </div>
</div>
            </details>


    </div>
</div>



Papermill parameters:

`depmap_modeling_df`: The path to the full DepMap modeling data set.


```python
DEPMAP_MODELING_DF: str = ""
```


```python
# Parameters
DEPMAP_MODELING_DF = "../modeling_data/depmap-modeling-data.csv"
```


```python
assert DEPMAP_MODELING_DF != "", "No path provided for the modeling data."
```


```python
depmap_modeling_df_path = Path(DEPMAP_MODELING_DF)

if not depmap_modeling_df_path.exists():
    raise FileNotFoundError(f"Could not find '{str(depmap_modeling_df_path)}'")
```


```python
pd.read_csv(depmap_modeling_df_path, low_memory=False, nrows=200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>p_dna_batch</th>
      <th>genome_alignment</th>
      <th>hugo_symbol</th>
      <th>screen</th>
      <th>multiple_hits_on_gene</th>
      <th>sgrna_target_chr</th>
      <th>sgrna_target_pos</th>
      <th>...</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAACCTGCGGCGGTCGCCA</td>
      <td>OVR3_c905R1</td>
      <td>-0.299958</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr8_66505451_-</td>
      <td>VXN</td>
      <td>sanger</td>
      <td>True</td>
      <td>8</td>
      <td>66505451</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.139595</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AACAGCACACCGGCCCCGT</td>
      <td>OVR3_c905R1</td>
      <td>0.267092</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_156009834_-</td>
      <td>IL9R</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>156009834</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACCTCCGGACTCCTCAGC</td>
      <td>OVR3_c905R1</td>
      <td>0.550477</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr7_39609658_-</td>
      <td>YAE1</td>
      <td>sanger</td>
      <td>True</td>
      <td>7</td>
      <td>39609658</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.923715</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACTCAAACTGACGCCGAA</td>
      <td>OVR3_c905R1</td>
      <td>-0.391922</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr1_117623388_-</td>
      <td>TENT5C</td>
      <td>sanger</td>
      <td>True</td>
      <td>1</td>
      <td>117623388</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AACTGACCTTGAAACGCTG</td>
      <td>OVR3_c905R1</td>
      <td>-1.562577</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr16_66933623_+</td>
      <td>CIAO2B</td>
      <td>sanger</td>
      <td>True</td>
      <td>16</td>
      <td>66933623</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>TGAGCTGGCAATGCTAGAT</td>
      <td>OVR3_c905R1</td>
      <td>0.565344</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_155774116_-</td>
      <td>SPRY3</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>155774116</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>196</th>
      <td>TGATGGAGCGAATCAGATG</td>
      <td>OVR3_c905R1</td>
      <td>-0.204959</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr16_66934065_+</td>
      <td>CIAO2B</td>
      <td>sanger</td>
      <td>True</td>
      <td>16</td>
      <td>66934065</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>197</th>
      <td>TGCACTTATGTGTGCCGCC</td>
      <td>OVR3_c905R1</td>
      <td>0.650650</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_156003692_-</td>
      <td>IL9R</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>156003692</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>198</th>
      <td>TGCTAGGACCCAACTGAGC</td>
      <td>OVR3_c905R1</td>
      <td>-0.517796</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr10_46580364_+</td>
      <td>SYT15</td>
      <td>sanger</td>
      <td>True</td>
      <td>10</td>
      <td>46580364</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.752471</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>199</th>
      <td>TGGAAAGTTGCCTCGTCCG</td>
      <td>OVR3_c905R1</td>
      <td>-0.218348</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr1_117622978_-</td>
      <td>TENT5C</td>
      <td>sanger</td>
      <td>True</td>
      <td>1</td>
      <td>117622978</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 25 columns</p>
</div>




```python
depmap_modeling_df = dd.read_csv(
    depmap_modeling_df_path,
    dtype={
        "age": "float64",
        "p_dna_batch": "object",
        "primary_or_metastasis": "object",
        "counts_final": "float64",
        "counts_initial": "float64",
    },
    low_memory=False,
)
```


```python
depmap_modeling_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>p_dna_batch</th>
      <th>genome_alignment</th>
      <th>hugo_symbol</th>
      <th>screen</th>
      <th>multiple_hits_on_gene</th>
      <th>sgrna_target_chr</th>
      <th>sgrna_target_pos</th>
      <th>...</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAACCTGCGGCGGTCGCCA</td>
      <td>OVR3_c905R1</td>
      <td>-0.299958</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr8_66505451_-</td>
      <td>VXN</td>
      <td>sanger</td>
      <td>True</td>
      <td>8</td>
      <td>66505451</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.139595</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AACAGCACACCGGCCCCGT</td>
      <td>OVR3_c905R1</td>
      <td>0.267092</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_156009834_-</td>
      <td>IL9R</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>156009834</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACCTCCGGACTCCTCAGC</td>
      <td>OVR3_c905R1</td>
      <td>0.550477</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr7_39609658_-</td>
      <td>YAE1</td>
      <td>sanger</td>
      <td>True</td>
      <td>7</td>
      <td>39609658</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.923715</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACTCAAACTGACGCCGAA</td>
      <td>OVR3_c905R1</td>
      <td>-0.391922</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr1_117623388_-</td>
      <td>TENT5C</td>
      <td>sanger</td>
      <td>True</td>
      <td>1</td>
      <td>117623388</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AACTGACCTTGAAACGCTG</td>
      <td>OVR3_c905R1</td>
      <td>-1.562577</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr16_66933623_+</td>
      <td>CIAO2B</td>
      <td>sanger</td>
      <td>True</td>
      <td>16</td>
      <td>66933623</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
depmap_modeling_df.columns
```




    Index(['sgrna', 'replicate_id', 'lfc', 'p_dna_batch', 'genome_alignment',
           'hugo_symbol', 'screen', 'multiple_hits_on_gene', 'sgrna_target_chr',
           'sgrna_target_pos', 'depmap_id', 'counts_final', 'counts_initial',
           'rna_expr', 'num_mutations', 'any_deleterious', 'any_tcga_hotspot',
           'any_cosmic_hotspot', 'is_mutated', 'copy_number', 'lineage',
           'lineage_subtype', 'primary_or_metastasis', 'is_male', 'age'],
          dtype='object')



## Basic checks


```python
FAILED_CHECKS = 0
```

Check that specific columns exist (prevents some really bonehead discoveries later on...).


```python
cols_that_should_exist = [
    "depmap_id",
    "sgrna",
    "hugo_symbol",
    "lfc",
    "screen",
    "num_mutations",
    "is_mutated",
    "lineage",
    "counts_final",
    "p_dna_batch",
    "primary_or_metastasis",
]

missing_cols = [
    col for col in cols_that_should_exist if col not in depmap_modeling_df.columns
]
if len(missing_cols) != 0:
    print(f"Some columns ({len(missing_cols)}) that should be present are not 😦")
    print("  missing columns: " + ", ".join(missing_cols))
    FAILED_CHECKS += 1
```

Check that specific columns have no missing (`NA`) values.


```python
cols_without_na = [
    "depmap_id",
    "sgrna",
    "hugo_symbol",
    "lfc",
    "screen",
    "num_mutations",
    "is_mutated",
    "lineage",
]

na_checks = depmap_modeling_df.isna()[cols_without_na].any().compute()
num_missed_checks = na_checks.sum()

if num_missed_checks > 0:
    FAILED_CHECKS += num_missed_checks
    print(na_checks[na_checks])
```


```python
na_checks
```




    depmap_id        False
    sgrna            False
    hugo_symbol      False
    lfc              False
    screen           False
    num_mutations    False
    is_mutated       False
    lineage          False
    dtype: bool



Check that all combinations of cell line, sgRNA, and experimental replicate only appear once.


```python
grp_cols = ["depmap_id", "sgrna", "replicate_id"]
ct_df = (
    depmap_modeling_df.assign(n=1)[grp_cols + ["n"]]
    .groupby(grp_cols)
    .count()
    .query("n > 1")
    .compute()
)

if not ct_df.shape[0] == 0:
    print("There are some sgRNA with multiple targets.")
    print(ct_df.head(20))
    FAILED_CHECKS += 1
```


```python
if FAILED_CHECKS > 0:
    raise Exception(f"There were {FAILED_CHECKS} failed checks.")
```

---


```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-06-22

    Python implementation: CPython
    Python version       : 3.10.4
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-152.o2.rc.hms.harvard.edu

    Git branch: per-lineage

    dask  : 2022.6.0
    pandas: 1.4.2
    numpy : 1.22.4




```python

```
