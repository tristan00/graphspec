import logging
import time

import numpy as np
import pytest
import support
from registered import KEPT, LINKED, RUNNING

from conftest import SPLITS
from graphspec import REPRESENTATIONS, Representation
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.representations.registry import FAMILIES, families
from tuning.bank.bank import Deriver
from tuning.families.model_spec import DERIVES, derived
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("added")
SHIPPED = dict(structure=("coupling", "events", "mixed"), sample=("neighbors", "walks"), encode=())
ADDED = (LINKED, KEPT, RUNNING)


@pytest.fixture
def represented():
    for entry in ADDED:
        REPRESENTATIONS.register(entry.name, entry)
    yield
    for entry in ADDED:
        del REPRESENTATIONS.held[entry.name]


def test_a_representation_registered_in_each_family_completes_one_fit_on_cpu(
    bank, packed, represented, tmp_path, statement
):
    started = time.perf_counter()
    assert {family: tuple(sorted(names)) for family, names in families().items()} == {
        family: tuple(sorted(SHIPPED[family] + tuple(entry.name for entry in ADDED if entry.family == family)))
        for family in FAMILIES
    }
    with pytest.raises(ValueError, match="states its family as one of structure, sample, encode, found 'drawn'"):
        Representation("test.drawn", "drawn", *(lambda *arguments: None,) * 3, None, None, lambda *arguments: None)
    opened = Deriver(bank, "cpu", RESIDENT)
    stated = {DERIVES % entry.name: True for entry in ADDED}
    with pytest.raises(KeyError, match="the model states no %s" % (DERIVES % LINKED.name)):
        derived(dict(view="snapshot"))
    cfg = dict(
        support.configuration(
            opened.registry, dict(view="snapshot", hidden=4, **stated), {"time/history": 2}, 5, SPLITS
        ),
        epochs=1,
    )
    assert opened.identity(cfg)["groups"] == [entry.name for entry in ADDED]
    result = fit(bank, tmp_path, cfg, opened.blocks(0, 2, cfg), 16, time.perf_counter() + 30, state(statement))
    assert read_json(tmp_path / "result.json")["status"] == "complete" and result["timings"]
    arrays = opened.arrays(0, cfg)
    entity, linked = np.asarray(arrays["entity"]), np.asarray(arrays["linked"])
    assert len(arrays["kept"]) == len(arrays["running"]) == len(entity)
    assert 0 < np.asarray(arrays["kept"]).sum() < len(entity) and (linked >= 0).any()
    assert np.array_equal(entity[linked[linked >= 0]], entity[linked >= 0])
    LOG.info("registered families fit exit seconds=%.3f", time.perf_counter() - started)
