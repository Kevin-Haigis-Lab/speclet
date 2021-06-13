```python
%load_ext autoreload
%autoreload 2
```

```python
import re
import string
import warnings
from pathlib import Path
from time import time

import arviz as az
import janitor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from theano import tensor as tt
```

```python
import src.modeling.simulation_based_calibration_helpers as sbc
from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.io import cache_io
from src.loggers import logger
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.models.speclet_six import SpecletSix
from src.plot.color_pal import FitMethodColors, ModelColors, SeabornColor
from src.project_enums import ModelFitMethod
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(
    gg.theme_bw()
    + gg.theme(
        figure_size=(4, 4),
        axis_ticks_major=gg.element_blank(),
        strip_background=gg.element_blank(),
    )
)
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 300
np.random.seed(RANDOM_SEED)

HDI_PROB = 0.89
```

```python
sp6 = SpecletSix("expt-sbc", cache_io.default_cache_dir(), debug=True)
```

```python
# sp6.cache_manager.clear_all_caches()
sp6.cache_manager.mcmc_cache_exists()
```

    False

```python
sbc_dir = results_path = sp6.cache_manager.cache_dir / "sbc"
```

```python
sp6.run_simulation_based_calibration(
    sbc_dir,
    fit_method=ModelFitMethod.mcmc,
    random_seed=RANDOM_SEED,
    size="small",
)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[06/12/21 13:48:37] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Generating mock data of size <span style="color: #008000; text-decoration-color: #008000">'small'</span>. <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:305</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[06/12/21 13:48:38] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span> data transformations.      <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:129</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation: <span style="color: #008000; text-decoration-color: #008000">'_drop_sgrna</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">s_that_map_to_multiple_genes'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> sgRNA that map to multiple <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:249</span>
                             genes.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'_drop_missing_copynumber'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> data points with missing   <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:259</span>
                             copy number.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'centered_copynumber_by_cellline'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'copy_number_cellline'</span> column.          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:26</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'centered_copynumber_by_gene'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'copy_number_gene'</span> column.              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:44</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation: <span style="color: #008000; text-decoration-color: #008000">'zscale_rna_</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">expression_by_gene_and_lineage'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'rna_expr_gene_lineage'</span> column.         <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:62</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'convert_is_mutated_to_numeric'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Converting <span style="color: #008000; text-decoration-color: #008000">'is_mutated'</span> column to <span style="color: #008000; text-decoration-color: #008000">'int'</span>.       <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:81</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Calling `model_specification<span style="font-weight: bold">()</span>` method.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:172</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Beginning PyMC3 model specification.          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:271</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Getting Theano shared variables.              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:279</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Creating PyMC3 model.                         <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:318</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Still need to implement varying effect for    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:319</span>
                             source on batch effect.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[06/12/21 13:48:48] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Found <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> cell line lineages.                   <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:360</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[06/12/21 13:48:53] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span> data transformations.      <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:129</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation: <span style="color: #008000; text-decoration-color: #008000">'_drop_sgrna</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">s_that_map_to_multiple_genes'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> sgRNA that map to multiple <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:249</span>
                             genes.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'_drop_missing_copynumber'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> data points with missing   <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:259</span>
                             copy number.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'centered_copynumber_by_cellline'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'copy_number_cellline'</span> column.          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:26</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'centered_copynumber_by_gene'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'copy_number_gene'</span> column.              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:44</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation: <span style="color: #008000; text-decoration-color: #008000">'zscale_rna_</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">expression_by_gene_and_lineage'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Adding <span style="color: #008000; text-decoration-color: #008000">'rna_expr_gene_lineage'</span> column.         <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:62</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Applying transformation:              <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:116</span>
                             <span style="color: #008000; text-decoration-color: #008000">'convert_is_mutated_to_numeric'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Converting <span style="color: #008000; text-decoration-color: #008000">'is_mutated'</span> column to <span style="color: #008000; text-decoration-color: #008000">'int'</span>.       <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:81</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Updating the MCMC sampling parameters.        <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_six.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_six.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:491</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Beginning MCMC sampling.                    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:277</span>
</pre>

    /n/data1/hms/dbmi/park/Cook/speclet/src/modeling/pymc3_sampling_api.py:122: FutureWarning: In v4.0, pm.sample will return an `arviz.InferenceData` object instead of a `MultiTrace` by default. You can pass return_inferencedata=True or return_inferencedata=False to be safe and silence this warning.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, σ_σ, a_offset, σ_a, σ_σ_a, μ_a_offset, σ_μ_a, μ_μ_a, d_offset, σ_d, σ_σ_d, μ_d_offset, σ_μ_d, μ_μ_d, h_offset, σ_h, μ_h, j_offset, σ_j, μ_j]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='24000' class='' max='24000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [24000/24000 21:38<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 4_000 draw iterations (8_000 + 16_000 draws total) took 1299 seconds.
    The number of effective samples is smaller than 25% for some parameters.

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='16000' class='' max='16000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [16000/16000 02:40<00:00]
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[06/12/21 14:14:01] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Finished MCMC sampling - caching results.   <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:290</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Caching InferenceData to <span style="color: #008000; text-decoration-color: #008000">'/n/data1/hms/dbm</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/cache_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">cache_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:245</span>
                             <span style="color: #008000; text-decoration-color: #008000">i/park/Cook/speclet/models/speclet-six_exp</span>
                             <span style="color: #008000; text-decoration-color: #008000">t-sbc/mcmc/inference-data.nc'</span>.
</pre>

```python
pm.model_to_graphviz(sp6.model)
```

![svg](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_8_0.svg)

## Visualization of mock data

```python
mock_data = sp6.data_manager.get_data()
print(mock_data.shape)
mock_data.head()
```

    (150, 15)

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
      <th>depmap_id</th>
      <th>hugo_symbol</th>
      <th>lineage</th>
      <th>p_dna_batch</th>
      <th>screen</th>
      <th>sgrna</th>
      <th>copy_number</th>
      <th>rna_expr</th>
      <th>is_mutated</th>
      <th>lfc</th>
      <th>copy_number_cellline</th>
      <th>copy_number_gene</th>
      <th>rna_expr_gene_lineage</th>
      <th>copy_number_cellline</th>
      <th>copy_number_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cellline0</td>
      <td>gene0</td>
      <td>lineage0</td>
      <td>batch0</td>
      <td>screen0</td>
      <td>gene0_sgrna0</td>
      <td>0.932745</td>
      <td>7.090937</td>
      <td>0</td>
      <td>2.217510</td>
      <td>-0.090782</td>
      <td>-0.096112</td>
      <td>-0.742856</td>
      <td>-0.090782</td>
      <td>-0.096112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cellline0</td>
      <td>gene0</td>
      <td>lineage0</td>
      <td>batch0</td>
      <td>screen0</td>
      <td>gene0_sgrna1</td>
      <td>0.871185</td>
      <td>7.246290</td>
      <td>0</td>
      <td>2.508963</td>
      <td>-0.152342</td>
      <td>-0.157672</td>
      <td>0.755915</td>
      <td>-0.152342</td>
      <td>-0.157672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cellline0</td>
      <td>gene0</td>
      <td>lineage0</td>
      <td>batch0</td>
      <td>screen0</td>
      <td>gene0_sgrna2</td>
      <td>0.982572</td>
      <td>7.241752</td>
      <td>0</td>
      <td>2.528404</td>
      <td>-0.040954</td>
      <td>-0.046285</td>
      <td>0.712536</td>
      <td>-0.040954</td>
      <td>-0.046285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cellline1</td>
      <td>gene0</td>
      <td>lineage1</td>
      <td>batch0</td>
      <td>screen0</td>
      <td>gene0_sgrna0</td>
      <td>1.457007</td>
      <td>12.607144</td>
      <td>0</td>
      <td>1.347310</td>
      <td>0.433199</td>
      <td>0.428149</td>
      <td>-0.412557</td>
      <td>0.433199</td>
      <td>0.428149</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cellline1</td>
      <td>gene0</td>
      <td>lineage1</td>
      <td>batch0</td>
      <td>screen0</td>
      <td>gene0_sgrna1</td>
      <td>0.955416</td>
      <td>14.421207</td>
      <td>0</td>
      <td>1.415043</td>
      <td>-0.068392</td>
      <td>-0.073442</td>
      <td>0.595885</td>
      <td>-0.068392</td>
      <td>-0.073442</td>
    </tr>
  </tbody>
</table>
</div>

```python
(
    gg.ggplot(mock_data, gg.aes(x="lfc"))
    + gg.geom_density(fill="black", alpha=0.1)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(figure_size=(4, 2))
)
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_11_0.png)

    <ggplot: (8779993093768)>

```python
plot_cols = ["depmap_id", "hugo_symbol", "screen", "lineage"]
plot_sizes = [(4, 4), (6, 4), (2, 4), (3, 4)]
for col, figure_size in zip(plot_cols, plot_sizes):
    (
        gg.ggplot(mock_data, gg.aes(x=col, y="lfc"))
        + gg.geom_boxplot(outlier_alpha=0.0)
        + gg.geom_jitter(alpha=0.4)
        + gg.theme(figure_size=figure_size)
    ).draw()
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_12_0.png)

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_12_1.png)

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_12_2.png)

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_12_3.png)

```python
(
    gg.ggplot(mock_data, gg.aes(x="hugo_symbol", y="rna_expr"))
    + gg.geom_boxplot(outlier_alpha=0.0)
    + gg.geom_jitter(gg.aes(color="lineage"), alpha=0.4)
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.theme(figure_size=(6, 3))
)
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_13_0.png)

    <ggplot: (8779954239741)>

```python
(
    gg.ggplot(mock_data, gg.aes(x="copy_number"))
    + gg.geom_density(fill="black", alpha=0.1)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
)
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_14_0.png)

    <ggplot: (8779956694414)>

## Model parameters

```python
sbc_manager = sbc.SBCFileManager(sbc_dir)

if sbc_manager.all_data_exists():
    sbc_res = sbc_manager.get_sbc_results()
else:
    FileNotFoundError("Could not locate SBC results data.")
```

```python
sbc_res.priors["μ_a"]
```

    array([[1.32751744, 1.43800259, 1.33298368, 1.38107235, 1.53012437,
            1.39844757, 1.42569601, 1.57365439, 1.46819278, 1.43234349]])

```python
(
    sbc_res.posterior_summary.reset_index(drop=False)
    .filter_string("parameter", "^μ_a\\[")
    .filter_string("parameter", "offset", complement=True)
    .reset_index(drop=True)
    .assign(real_value=sbc_res.priors["μ_a"].flatten())
)
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
      <th>parameter</th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>real_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>μ_a[0]</td>
      <td>-0.011</td>
      <td>0.928</td>
      <td>-1.474</td>
      <td>1.460</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25771.0</td>
      <td>10986.0</td>
      <td>1.0</td>
      <td>1.327517</td>
    </tr>
    <tr>
      <th>1</th>
      <td>μ_a[1]</td>
      <td>-0.006</td>
      <td>0.928</td>
      <td>-1.490</td>
      <td>1.461</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>26223.0</td>
      <td>11497.0</td>
      <td>1.0</td>
      <td>1.438003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>μ_a[2]</td>
      <td>0.058</td>
      <td>0.929</td>
      <td>-1.421</td>
      <td>1.522</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25809.0</td>
      <td>11502.0</td>
      <td>1.0</td>
      <td>1.332984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>μ_a[3]</td>
      <td>-0.099</td>
      <td>0.933</td>
      <td>-1.539</td>
      <td>1.427</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25562.0</td>
      <td>11388.0</td>
      <td>1.0</td>
      <td>1.381072</td>
    </tr>
    <tr>
      <th>4</th>
      <td>μ_a[4]</td>
      <td>-0.014</td>
      <td>0.945</td>
      <td>-1.536</td>
      <td>1.439</td>
      <td>0.006</td>
      <td>0.009</td>
      <td>26089.0</td>
      <td>11379.0</td>
      <td>1.0</td>
      <td>1.530124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>μ_a[5]</td>
      <td>-0.012</td>
      <td>0.948</td>
      <td>-1.526</td>
      <td>1.467</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25813.0</td>
      <td>11782.0</td>
      <td>1.0</td>
      <td>1.398448</td>
    </tr>
    <tr>
      <th>6</th>
      <td>μ_a[6]</td>
      <td>-0.015</td>
      <td>0.948</td>
      <td>-1.569</td>
      <td>1.439</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25676.0</td>
      <td>11414.0</td>
      <td>1.0</td>
      <td>1.425696</td>
    </tr>
    <tr>
      <th>7</th>
      <td>μ_a[7]</td>
      <td>-0.015</td>
      <td>0.944</td>
      <td>-1.509</td>
      <td>1.474</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25859.0</td>
      <td>11854.0</td>
      <td>1.0</td>
      <td>1.573654</td>
    </tr>
    <tr>
      <th>8</th>
      <td>μ_a[8]</td>
      <td>-0.014</td>
      <td>0.947</td>
      <td>-1.578</td>
      <td>1.419</td>
      <td>0.006</td>
      <td>0.008</td>
      <td>25076.0</td>
      <td>11282.0</td>
      <td>1.0</td>
      <td>1.468193</td>
    </tr>
    <tr>
      <th>9</th>
      <td>μ_a[9]</td>
      <td>-0.014</td>
      <td>0.947</td>
      <td>-1.563</td>
      <td>1.451</td>
      <td>0.006</td>
      <td>0.009</td>
      <td>25229.0</td>
      <td>11456.0</td>
      <td>1.0</td>
      <td>1.432343</td>
    </tr>
  </tbody>
</table>
</div>

**BUG**: Distribution should be skewed negative.

```python
sns.displot(sbc_res.priors["lfc"].flatten());
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_20_0.png)

```python
x = sbc_res.inference_obj.posterior["a"][:, :, 0].values.flatten()
y = sbc_res.inference_obj.posterior["d"][:, :, 0].values.flatten()
# y = sbc_res.inference_obj.posterior["d"][:, :, 0].values.flatten()
sns.scatterplot(x, y)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.





    <AxesSubplot:>

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_21_2.png)

```python
az.plot_trace(sbc_res.inference_obj, var_names=["μ_μ_a", "μ_μ_d"], compact=False);
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_22_0.png)

```python
az.plot_forest(
    sbc_res.inference_obj,
    var_names=["σ_h", "μ_h", "σ_j", "μ_j", "μ_μ_d", "μ_μ_a", "σ_σ_d"],
);
```

![png](020_010_experimentation-speclet6-sbc_files/020_010_experimentation-speclet6-sbc_23_0.png)

```python
az.summary(sbc_res.inference_obj, var_names=["σ_d"], hdi_prob=HDI_PROB).assign(
    real_value=sbc_res.priors["σ_d"].flatten()
)
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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>real_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>σ_d[0]</th>
      <td>0.202</td>
      <td>0.238</td>
      <td>0.0</td>
      <td>0.452</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6568.0</td>
      <td>6166.0</td>
      <td>1.0</td>
      <td>0.228256</td>
    </tr>
    <tr>
      <th>σ_d[1]</th>
      <td>0.264</td>
      <td>0.308</td>
      <td>0.0</td>
      <td>0.593</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5988.0</td>
      <td>6770.0</td>
      <td>1.0</td>
      <td>0.048012</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

```python

```

```python

```

```python

```

```python

```

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```
