"""Utilities for snakemake workflow: 010_010_run-crc-sampling-snakemake."""


# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
sample_models_memory_lookup = {
    "crc_ceres_mimic": {
        True: {"ADVI": 15, "MCMC": 20},
        False: {"ADVI": 20, "MCMC": 40},
    },
    "speclet_two": {True: {"ADVI": 7, "MCMC": 30}, False: {"ADVI": 30, "MCMC": 150}},
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
sample_models_time_lookup = {
    "crc_ceres_mimic": {
        True: {"ADVI": "00:30:00", "MCMC": "00:30:00"},
        False: {"ADVI": "03:00:00", "MCMC": "06:00:00"},
    },
    "speclet_two": {
        True: {"ADVI": "00:30:00", "MCMC": "12:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
    "speclet_three": {
        True: {"ADVI": "00:30:00", "MCMC": "12:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
    "speclet_four": {
        True: {"ADVI": "00:30:00", "MCMC": "12:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
}


def is_debug(name: str) -> bool:
    """Determine the debug status of model name."""
    return "debug" in name


def get_from_lookup(w, lookup_dict):
    """Generic dictionary lookup for the params in the `sample_models` step."""
    return lookup_dict[w.model][is_debug(w.model_name)][w.fit_method]


def get_sample_models_memory(w) -> int:
    """Memory required for the `sample_models` step."""
    try:
        return get_from_lookup(w, sample_models_memory_lookup) * 1000
    except BaseException:
        if is_debug(w.model_name):
            return 7 * 1000
        else:
            return 20 * 1000


def get_sample_models_time(w) -> str:
    """Time required for the `sample_models` step."""
    try:
        return get_from_lookup(w, sample_models_time_lookup)
    except BaseException:
        if is_debug(w.model_name):
            return "00:30:00"
        else:
            return "01:00:00"


def get_sample_models_partition(w) -> str:
    """Time O2 partition for the `sample_models` step."""
    t = [int(x) for x in get_sample_models_time(w).split(":")]
    total_minutes = (t[0] * 60) + t[1]
    if total_minutes <= (12 * 60):
        return "short"
    elif total_minutes <= (5 * 24 * 60):
        return "medium"
    else:
        return "long"
