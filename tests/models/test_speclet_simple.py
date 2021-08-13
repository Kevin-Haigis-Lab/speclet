from pathlib import Path
from typing import Optional

import faker
import pytest

from src.managers import model_data_managers as dms
from src.models.speclet_simple import SpecletSimple

fake = faker.Faker()


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize(
    "data_manager", [None, dms.CrcDataManager, dms.MockDataManager]
)
def test_init(tmp_path: Path, debug: bool, data_manager: Optional[dms.DataManager]):
    sps = SpecletSimple(fake.name(), tmp_path, debug, data_manager)
    assert isinstance(sps, SpecletSimple)
    assert sps.data_manager is not None
    assert sps.debug == debug
    assert sps.cache_manager is not None
    assert sps.mcmc_results is None
    assert sps.advi_results is None
