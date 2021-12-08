"""Various functions used in many test modules."""

from typing import Any, Callable, Optional

from pydantic import BaseModel

from speclet.modeling import pymc3_helpers as pmhelp
from speclet.models.speclet_model import SpecletModel
from speclet.project_enums import ModelParameterization as MP

#### ---- General ---- ####


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Take any arguments and do nothing.

    Returns:
        None: None
    """
    return None


#### ---- src.models ---- ####


def assert_changing_configuration_resets_model(
    sp_model: SpecletModel, new_config: BaseModel, default_config: BaseModel
) -> None:
    """Test that changing a speclet model's config resets the model.

    Also checks that if the configuration file does not change, then the model should
    not reset.

    Args:
        sp_model (SpecletModel): Speclet model instance.
        new_config (BaseModel): New configuration file.
        default_config (BaseModel): The default configuration file.
    """
    assert sp_model.model is None
    sp_model.build_model()
    assert sp_model.model is not None
    sp_model.set_config(new_config.dict())
    if new_config == default_config:
        assert sp_model.model is not None
    else:
        assert sp_model.model is None


PreCheckCallback = Callable[[str, str], bool]


def assert_model_reparameterization(
    sp_model: SpecletModel,
    config: BaseModel,
    pre_check_callback: Optional[PreCheckCallback] = None,
) -> None:
    """Assert that parameterizations in the config result in changes in the model.

    Args:
        sp_model (SpecletModel): Speclet model instance.
        config (BaseModel): Configuration file.
        pre_check_callback (Optional[PreCheckCallback], optional): A callable object
          (e.g. function) that is called before each parameterization check. If it
          returns `True`, the check is skipped. Defaults to None.
    """
    assert sp_model.model is None
    sp_model.build_model()
    assert sp_model.model is not None
    rv_names = pmhelp.get_random_variable_names(sp_model.model)
    unobs_names = pmhelp.get_deterministic_variable_names(sp_model.model)
    for param_name, param_method in config.dict().items():
        if pre_check_callback is not None and pre_check_callback(
            param_name, param_method
        ):
            continue
        assert (param_name in set(rv_names)) == (param_method is MP.CENTERED)
        assert (param_name in set(unobs_names)) == (param_method is MP.NONCENTERED)


# ---- Comparing dictionaries ----


def assert_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    """Compare dictionaries.

    Compares the two dictionaries using the keys from `d1` only.

    Args:
        d1 (dict[str, Any]): Dictionary one.
        d2 (dict[str, Any]): Dictionary two.
    """
    for k, v in d1.items():
        if isinstance(v, dict):
            assert_dicts(v, d2.get(k, {}))
        else:
            assert v == d2[k]
    return None
