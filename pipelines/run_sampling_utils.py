"""Utilities for snakemake workflow: 010_010_run-crc-sampling-snakemake."""


# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
sample_models_memory_lookup = {
    "speclet-test-model": {
        True: {"ADVI": 8, "MCMC": 8},
        False: {"ADVI": 8, "MCMC": 8},
    },
    "crc-ceres-mimic": {
        True: {"ADVI": 15, "MCMC": 20},
        False: {"ADVI": 20, "MCMC": 40},
    },
    "speclet-two": {True: {"ADVI": 7, "MCMC": 30}, False: {"ADVI": 40, "MCMC": 150}},
    "speclet-three": {True: {"ADVI": 7, "MCMC": 60}, False: {"ADVI": 40, "MCMC": 150}},
    "speclet-four": {True: {"ADVI": 7, "MCMC": 60}, False: {"ADVI": 40, "MCMC": 150}},
    "speclet-five": {True: {"ADVI": 7, "MCMC": 60}, False: {"ADVI": 40, "MCMC": 150}},
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
sample_models_time_lookup = {
    "speclet-test-model": {
        True: {"ADVI": "00:05:00", "MCMC": "00:05:00"},
        False: {"ADVI": "00:10:00", "MCMC": "00:10:00"},
    },
    "crc-ceres-mimic": {
        True: {"ADVI": "00:30:00", "MCMC": "00:30:00"},
        False: {"ADVI": "03:00:00", "MCMC": "06:00:00"},
    },
    "speclet-two": {
        True: {"ADVI": "00:30:00", "MCMC": "08:00:00"},
        False: {"ADVI": "12:00:00", "MCMC": "48:00:00"},
    },
    "speclet-three": {
        True: {"ADVI": "00:30:00", "MCMC": "24:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
    "speclet-four": {
        True: {"ADVI": "03:00:00", "MCMC": "24:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
    "speclet-five": {
        True: {"ADVI": "03:00:00", "MCMC": "24:00:00"},
        False: {"ADVI": "10:00:00", "MCMC": "48:00:00"},
    },
}


def is_debug(name: str) -> bool:
    """Determine the debug status of model name."""
    return "debug" in name


def get_from_lookup(w, lookup_dict, fit_method: str):
    """Generic dictionary lookup for the params in the `sample_models` step."""
    return lookup_dict[w.model][is_debug(w.model_name)][fit_method]


def get_sample_models_memory(w, fit_method: str) -> int:
    """Memory required for the `sample_models` step."""
    try:
        return (
            get_from_lookup(w, sample_models_memory_lookup, fit_method=fit_method)
            * 1000
        )
    except BaseException:
        if is_debug(w.model_name):
            return 7 * 1000
        else:
            return 20 * 1000


def get_sample_models_time(w, fit_method: str) -> str:
    """Time required for the `sample_models` step."""
    try:
        return get_from_lookup(w, sample_models_time_lookup, fit_method=fit_method)
    except BaseException:
        if is_debug(w.model_name):
            return "00:30:00"
        else:
            return "01:00:00"


def get_sample_models_partition(w, fit_method: str) -> str:
    """Time O2 partition for the `sample_models` step."""
    t = [int(x) for x in get_sample_models_time(w, fit_method=fit_method).split(":")]
    total_minutes = (t[0] * 60) + t[1]
    if total_minutes <= (12 * 60):
        return "short"
    elif total_minutes <= (5 * 24 * 60):
        return "medium"
    else:
        return "long"


def cli_is_debug(w):
    return "--debug" if "debug" in w.model_name else "--no-debug"
