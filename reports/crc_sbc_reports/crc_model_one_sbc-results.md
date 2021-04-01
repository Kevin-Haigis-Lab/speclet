# Model Report

```python
import warnings
from pathlib import Path
from time import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns

from src.context_managers import set_directory
from src.data_processing import common as dphelp
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling import sampling_pymc3_models as sampling
from src.modeling.sampling_pymc3_models import SamplingArguments
from src.plot.color_pal import SeabornColor

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("..", "models", "modeling_cache", "pymc3_model_cache")
```

Parameters for papermill:

- `MODEL`: which model was tested

## Setup

### Papermill parameters

```python
MODEL = ""
```

```python
# Parameters
MODEL = "crc_model_one"

```

```python
# speclet_model = sampling.sample_speclet_model(
#     MODEL, name=MODEL_NAME, debug=DEBUG, random_seed=RANDOM_SEED, touch=False
# )
```

```python
# model_res = speclet_model.advi_results
```

```python
import os

os.getcwd()
```

    '/n/data2/dfci/cancerbio/haigis/Cook/speclet/reports/crc_sbc_reports'

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

    execution time: 0.00 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-04-01

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.21.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-163.o2.rc.hms.harvard.edu

    Git branch: simulation-based-calibration

    matplotlib: 3.3.4
    arviz     : 0.11.2
    pandas    : 1.2.3
    numpy     : 1.20.1
    plotnine  : 0.7.1
    pymc3     : 3.11.1
    seaborn   : 0.11.1
