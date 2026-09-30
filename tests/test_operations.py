import logging
import time
from pathlib import Path

import fixture
import keyed
import numpy as np
import pytest
import support

from graphspec import OPERATIONS, Operation, Phase
from graphspec.budgets import GraphLimits
from graphspec.engine import segments, transform
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import parameters as specs
from graphspec.operations.nodes import NODES
from graphspec.operations.operation import listed
from graphspec.operations.pipeline import resolved
from graphspec.operations.window import WINDOW
from graphspec.views.graph import DENSE, Dense
from tuning.bank.bank import Deriver

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("operations")
SHIFT = specs.number("test/shift", 0, 4, 1)
OPERATED = {
    "02_periods_sample_coupling": dict(
        stage_3_nodes="e6f9e7752a4d6e9d",
        stage_4_derived_relations="f22576c189d752dc",
        stage_5_relations="f22576c189d752dc",
        stage_6_edge_filters="f22576c189d752dc",
        stage_7_relation_merging="f22576c189d752dc",
    ),
    "05_sample_drop_neighbors": dict(
        stage_3_nodes="48d24f17b5931ed2",
        stage_4_derived_relations="c2dcfe789d93bcc7",
        stage_5_relations="c2dcfe789d93bcc7",
        stage_6_edge_filters="9805b03cc6068921",
        stage_7_relation_merging="a258779988471d56",
    ),
    "06_project_complete_walks": dict(
        stage_4_derived_relations="a0aa6fadad7ea7a7",
        stage_5_relations="23a434263df17f37",
        stage_6_edge_filters="b0444a588d12dc19",
        stage_7_relation_merging="2447fdee246af294",
    ),
    "07_periods_hops_ppr_coupling": dict(
        stage_4_derived_relations="252100c0f13eb1a4",
        stage_5_relations="6b4890f3ab4d1317",
        stage_6_edge_filters="6b4890f3ab4d1317",
        stage_7_relation_merging="6b4890f3ab4d1317",
    ),
    "08_heat_sdrf_untyped_events": dict(
        stage_4_derived_relations="338b837255b54882",
        stage_5_relations="5a343dffbea92b49",
        stage_6_edge_filters="5a343dffbea92b49",
        stage_7_relation_merging="834608be78db1971",
    ),
    "09_fosr_drop_cap_walks": dict(
        stage_5_relations="d16ae7e1c28d6021",
        stage_6_edge_filters="f86dff3aba8d8247",
        stage_7_relation_merging="26bc1193698acc32",
    ),
}


def shifted(graph, registry, settings, seed, limits, carry, held):
    return graph.replace(period=(graph.period + settings["test/shift"]).astype(graph.period.dtype))


def stray(graph, registry, settings, seed, limits, carry, held):
    return graph.replace(period=graph.period + settings["time/window"])


def stated(registry):
    return (SHIFT,)


def made(name, function):
    return Operation("test", name, (Phase("apply", function, None, None),), stated, (), (), (), ("time",), 2)


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
        phases=[dict(name="apply", carries=False, parts=False)],
    )
    assert [phase["name"] for phase in found["graphspec.derive"]["phases"]] == ["begin", "each", "finish", "assemble"]
    assert [(phase["carries"], phase["parts"]) for phase in found["graphspec.edges"]["phases"]] == [
        (True, True),
        (True, False),
    ]
    assert set(support.SEVEN) <= set(found)
    assert [parameter["name"] for parameter in found["graphspec.periods"]["parameters"]] == [
        "time/periods",
        "time/period_length",
    ]
    assert found["graphspec.derive"]["reads"] == [] and found["graphspec.edges"]["reads"] == ["graphspec.derive"]
    assert {name for name in found if name.startswith("graphspec.")} == set(support.SEVEN)
    with pytest.raises(ValueError, match="the operation test.shift is registered twice"):
        OPERATIONS.register("test.shift", made("shift", shifted))
    with pytest.raises(ValueError, match="'test.shift' is registered as test.moved"):
        OPERATIONS.register("test.moved", made("shift", shifted))
    with pytest.raises(ValueError, match="states its name as a word without a dot, found 'a.b'"):
        made("a.b", shifted)
    dense = support.shown(held, 0).dense()
    steps = resolved(registry, [dict(operation="test.shift", parameters={"test/shift": 3})])
    moved = segments.whole(dense, registry, steps, transform.SEED, GraphLimits("cpu"), None)
    assert np.array_equal(moved.period, dense.period + 3) and moved.obs_step is dense.obs_step
    carry, opened = segments.opening(dense)
    with pytest.raises(KeyError, match="the operation test.stray reads time/window, which is none of the parameters"):
        OPERATIONS.named("test.stray")(
            opened, registry, {"test/shift": 3}, transform.SEED, GraphLimits("cpu"), carry, {}, None
        )
    lacking = {"time/scheme": "window", "time/unit": "step"}
    with pytest.raises(KeyError, match="the operation graphspec.window reads time/window, which the spec does not"):
        WINDOW(
            opened, registry, lacking, transform.SEED, GraphLimits("cpu"), carry, WINDOW.initial(registry, lacking), None
        )
    kind = registry.nodes[0]
    nodes = support.pipelined(registry, {}, ("graphspec.nodes",))[0][0]["parameters"]
    idle = dict(nodes, **{"nodes/%s/keep" % kind: False})
    with pytest.raises(
        ValueError,
        match="the parameter nodes/%s/filter, read by the operation graphspec.nodes, is inactive under its "
        "condition and stated by the spec" % kind,
    ):
        NODES(opened, registry, idle, transform.SEED, GraphLimits("cpu"), carry, NODES.initial(registry, nodes), None)
    LOG.info("registered operation exit seconds=%.3f", time.perf_counter() - started)


def identical(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


@pytest.mark.parametrize("place", range(10))
def test_each_operation_called_alone_on_the_fixture_gives_the_arrays_of_its_stage(ground, grounded, place):
    started = time.perf_counter()
    held, _ = fixture.configurations(ground)
    entry, labels = held[place], fixture.labels(ground)
    opened = Deriver(grounded[entry["source"]["name"]] / "bank", "cpu", RESIDENT)
    registry = support.fitted(opened)
    steps = resolved(registry, fixture.pipeline(opened.registry, entry)[0])
    names = [step.name for step in steps]
    assert len(names) == len(labels) and fixture.holds(names, len(names))
    level, folder = entry["source"]["level"], ground / entry["folder"]
    stored = [fixture.arrays(folder / name, level) for name in [entry["files"]["input"]] + entry["files"]["stages"]]
    carried, found = {}, {}
    for index, step in enumerate(steps):
        label = labels[support.SEVEN.index(step.name)]
        stage = Path(entry["files"]["stages"][index]).stem
        replaced = fixture.drawn(entry, stage)
        for name in entry["graphs"]:
            if fixture.holds(names, index):
                given = Dense(name, [], len(stored[index][name]["period"]), **stored[index][name])
            else:
                given = carried[name]
            carry, given = segments.opening(given)
            before = {key: np.array(value) for key, value in given.arrays.items()}
            result, _ = step(given, registry, transform.SEED, GraphLimits("cpu"), carry, step.initial(registry), None)
            assert all(identical(before[key], given.arrays[key]) for key in DENSE), (label, name)
            if fixture.holds(names, index + 1):
                assert set(fixture.differing(result.arrays, stored[index + 1][name])) <= set(replaced), (label, name)
                if replaced:
                    found.setdefault(stage, []).append(keyed.pinned(result, registry, replaced))
            carried[name] = result
            copied = [
                key
                for key in DENSE
                if result.arrays[key] is not given.arrays[key] and identical(result.arrays[key], given.arrays[key])
            ]
            assert copied == [], (label, name)
            assert (result is given) == (label not in entry["stages_changing_arrays"]), (label, name)
    pins = {stage: keyed.combined(held) for stage, held in found.items()}
    assert pins == OPERATED.get(entry["name"], {})
    LOG.info("operations on the fixture exit name=%s seconds=%.3f", entry["name"], time.perf_counter() - started)
