import logging
import time

import fixture
import numpy as np
import pytest
import support

from graphspec import OPERATIONS, Operation
from graphspec.budgets import GraphLimits
from graphspec.engine import transform
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import parameters as specs
from graphspec.operations.operation import PIPELINE, listed
from graphspec.operations.window import WINDOW
from graphspec.views.graph import DENSE, Dense
from tuning.bank.bank import Deriver

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("operations")
SHIFT = specs.number("test/shift", 0, 4, 1)


def shifted(graph, registry, settings, seed, limits, carry):
    return graph.replace(period=(graph.period + settings["test/shift"]).astype(graph.period.dtype))


def stray(graph, registry, settings, seed, limits, carry):
    return graph.replace(period=graph.period + settings["time/window"])


def stated(registry):
    return (SHIFT,)


def made(name, function):
    return Operation("test", name, function, stated, (), (), (), ("time",), 2)


@pytest.fixture
def registered():
    found = [made("shift", shifted), made("stray", stray)]
    for operation in found:
        OPERATIONS.register(operation.name, operation)
    yield found
    for operation in found:
        del OPERATIONS.held[operation.name]


def test_an_operation_registered_by_a_test_is_listed_with_its_parameters_and_reads_no_other(bank, registered):
    started = time.perf_counter()
    held = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(held)
    found = {entry["name"]: entry for entry in listed(held.registry)}
    assert found["test.shift"] == dict(
        name="test.shift",
        version=2,
        parameters=[dict(name="test/shift", kind="int", default=1, low=0, high=4, log=False, options=[])],
        reads=[],
        splits=[],
        requires=[],
        changes=["time"],
    )
    assert set(PIPELINE) <= set(found)
    assert [parameter["name"] for parameter in found["graphspec.periods"]["parameters"]] == [
        "time/periods",
        "time/period_length",
    ]
    assert found["graphspec.derive"]["reads"] == ["graphspec.relations"]
    assert {name for name in found if name.startswith("graphspec.")} == set(PIPELINE)
    with pytest.raises(ValueError, match="the operation test.shift is registered twice"):
        OPERATIONS.register("test.shift", made("shift", shifted))
    with pytest.raises(ValueError, match="'test.shift' is registered as test.moved"):
        OPERATIONS.register("test.moved", made("shift", shifted))
    with pytest.raises(ValueError, match="states its name as a word without a dot, found 'a.b'"):
        made("a.b", shifted)
    spec = dict(specs.configured(specs.space(held.registry, ()), support.several(held.registry)), **{"test/shift": 3})
    dense = support.shown(held, 0).dense()
    stages = ((name, OPERATIONS.named(name)) for name in ("test.shift",))
    moved = transform.derive(dense, registry, spec, GraphLimits("cpu"), stages=tuple(stages))
    assert np.array_equal(moved.period, dense.period + 3) and moved.obs_step is dense.obs_step
    with pytest.raises(KeyError, match="the operation test.stray reads time/window, which is none of the parameters"):
        OPERATIONS.named("test.stray")(dense, registry, spec, transform.SEED, GraphLimits("cpu"), None)
    lacking = {name: value for name, value in spec.items() if name != "time/window"}
    with pytest.raises(KeyError, match="the operation graphspec.window reads time/window, which the spec does not"):
        WINDOW(dense, registry, lacking, transform.SEED, GraphLimits("cpu"), None)
    kind = registry.nodes[0]
    idle = dict(spec, **{"nodes/%s/keep" % kind: False})
    with pytest.raises(
        ValueError,
        match="the parameter nodes/%s/filter, read by the operation graphspec.nodes, is inactive under its "
        "condition and stated by the spec" % kind,
    ):
        OPERATIONS.named("graphspec.nodes")(dense, registry, idle, transform.SEED, GraphLimits("cpu"), None)
    LOG.info("registered operation exit seconds=%.3f", time.perf_counter() - started)


def identical(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


@pytest.mark.parametrize("place", range(10))
def test_each_operation_called_alone_on_the_fixture_gives_the_arrays_of_its_stage(ground, grounded, place):
    started = time.perf_counter()
    entry = fixture.configurations(ground)[0][place]
    opened = Deriver(grounded[entry["source"]["name"]] / "bank", "cpu", RESIDENT)
    registry = support.fitted(opened)
    spec = specs.configured(specs.space(opened.registry, tuple(entry["groups"])), entry["values"])
    level, folder = entry["source"]["level"], ground / entry["folder"]
    stored = [fixture.arrays(folder / name, level) for name in [entry["files"]["input"]] + entry["files"]["stages"]]
    for index, (label, operation) in enumerate(transform.STAGES):
        for name in entry["graphs"]:
            given = Dense(name, [], len(stored[index][name]["period"]), **stored[index][name])
            before = {key: np.array(value) for key, value in given.arrays.items()}
            result = operation(given, registry, spec, transform.SEED, GraphLimits("cpu"), None)
            assert all(identical(before[key], given.arrays[key]) for key in DENSE), (label, name)
            assert fixture.differing(result.arrays, stored[index + 1][name]) == [], (label, name)
            copied = [
                key
                for key in DENSE
                if result.arrays[key] is not given.arrays[key] and identical(result.arrays[key], given.arrays[key])
            ]
            assert copied == [], (label, name)
            assert (result is given) == (label not in entry["stages_changing_arrays"]), (label, name)
    LOG.info("operations on the fixture exit name=%s seconds=%.3f", entry["name"], time.perf_counter() - started)
