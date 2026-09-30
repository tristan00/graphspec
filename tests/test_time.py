import copy
import logging
import time

import numpy as np
import pyarrow as pa
import pytest
import support

from conftest import SPLITS, inline
from graphspec import FORMS, SEES, SEVERAL, STEPS, TIES
from graphspec.build import forms
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed as clocked
from graphspec.operations import parameters as specs
from graphspec.schema import time as clock
from graphspec.schema.schema import VERSION, Schema
from graphspec.sources import synthetic
from graphspec.views import visible
from graphspec.views.graph import Graph
from graphspec.views.view import sequences, view
from graphspec.views.visible import unstated
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
configuration = support.configuration
pytestmark = pytest.mark.xdist_group("time")
MAKERS, SITES = ("m0", "m1", "m2"), ("s0", "s1", "s2", "s3")
MOMENTS = (0, 2, 4, 6, 8, 10)
STRIDE = 2
SPLIT = ("train", "train", "train", "train", "validation", "validation")
SERVES = (("m0", "s0"), ("m1", "s1"), ("m2", "s2"), ("m0", "s3"))
BESIDE = (("s0", "s1"), ("s2", "s3"))
SHAPED = ("state", "event", "snapshot", "change", "signal")


@clocked("time.stated")
def stated(model, units, ties, observers):
    return dict(
        version=VERSION,
        time=dict(model=model, units=units, ties=ties),
        observers=observers,
        columns=dict(
            size=dict(type="number", encoder="value"),
            load=dict(type="number", encoder="value"),
            weight=dict(type="number", encoder="logarithm"),
            ask=dict(type="number", encoder="value"),
        ),
        graph=dict(columns=[]),
        nodes=dict(maker=dict(columns=["size"], holds=None), site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(
            serves=synthetic.edge("maker", "site", ["weight"]),
            beside=synthetic.edge("site", "site", [], directed=False),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(
                first=dict(sort="node", kinds=["maker"]),
                second=dict(sort="node", kinds=["site"]),
                third=dict(sort="node", kinds=["site"]),
            ),
            kinds={
                kind: synthetic.kind(
                    ["first", "second", "third"],
                    ["ask"],
                    dict(kind="number", series="outcome", about="first"),
                    ["first", "second"],
                )
                for kind in ("plain", "urgent")
            },
        ),
    )


@clocked("time.statement")
def statement(form, parameters, rows, **named):
    return dict(named, form=form, parameters=parameters, rows=rows)


def level(name, place, moment):
    return float(1 + place + 0.5 * moment + 0.25 * len(name))


@clocked("time.timed")
def timed(form, moments):
    if form == "signal":
        return dict(index=[moment // STRIDE for moment in moments]), dict(start=0, stride=STRIDE)
    if form == "snapshot":
        return dict(at=list(moments)), dict(times=list(MOMENTS))
    return dict(at=list(moments)), dict(keys=["key"]) if form == "change" else {}


@clocked("time.varying")
def varying(form, name, keys, column, kind):
    rows = dict(key=[], **{column: []})
    moments = []
    for place, key in enumerate(keys):
        rows["key"] += [key] * len(MOMENTS)
        rows[column] += [level(name, place, moment) for moment in MOMENTS]
        moments += list(MOMENTS)
    if form == "state":
        rows.update(since=moments, until=[None if m == MOMENTS[-1] else m + STRIDE for m in moments])
        return statement("state", {}, rows, kind=kind)
    when, parameters = timed(form, moments)
    return statement(form, parameters, dict(rows, **when), kind=kind)


@clocked("time.joined")
def joined(form, name):
    serving = dict(source=[], target=[], present=[], weight=[])
    moments = []
    for place, (source, target) in enumerate(SERVES):
        held = MOMENTS[place % 2 :]
        serving["source"] += [source] * len(held)
        serving["target"] += [target] * len(held)
        serving["present"] += [True] * len(held)
        serving["weight"] += [level(name, place, moment) for moment in held]
        moments += list(held)
    near = dict(source=[pair[0] for pair in BESIDE], target=[pair[1] for pair in BESIDE])
    near["present"] = [True] * len(BESIDE)
    ends = dict(serves=("maker", "site"), beside=("site", "site"))
    if form in ("state", "signal"):
        serving.update(since=moments, until=[None if m == MOMENTS[-1] else m + STRIDE for m in moments])
        near.update(since=[0] * len(BESIDE), until=[None] * len(BESIDE))
        return [
            statement("state", {}, serving, table="serves", ends=ends["serves"]),
            statement("state", {}, near, table="beside", ends=ends["beside"]),
        ]
    parameters = dict(change=dict(keys=["source", "target"]), snapshot=dict(times=list(MOMENTS)), event={})
    serving["at"] = moments
    if form == "change":
        near["at"] = [0] * len(BESIDE)
    else:
        near = {key: [value for _ in MOMENTS for value in values] for key, values in near.items()}
        near["at"] = [moment for moment in MOMENTS for _ in BESIDE]
    return [
        statement(form, parameters[form], serving, table="serves", ends=ends["serves"]),
        statement(form, parameters[form], near, table="beside", ends=ends["beside"]),
    ]


@clocked("time.asked")
def asked(name, moments, observers):
    count = len(moments)
    tasks = dict(
        key=[100 + place for place in range(count)],
        kind=["plain" if place % 2 else "urgent" for place in range(count)],
        split=[SPLIT[place % len(SPLIT)] for place in range(count)],
        weight=[None] * count,
        ask=[float(place) for place in range(count)],
        references=dict(
            first=[MAKERS[place % len(MAKERS)] for place in range(count)],
            second=[SITES[place % 3] for place in range(count)],
            third=[SITES[(place + 1) % 3] for place in range(count)],
        ),
        target=[level(name, place % len(MAKERS), 0) + 0.1 * place for place in range(count)],
        known=[None] * count,
    )
    if moments[0] is not None:
        tasks["time"] = list(moments)
    if observers is not None:
        tasks["observer"] = list(observers)
    return tasks


@clocked("time.graph")
def graph(name, unit, values, edges, moments, observers):
    return dict(
        name=name,
        unit=unit,
        columns={},
        nodes=dict(maker=list(MAKERS), site=list(SITES)),
        values=values,
        edges=edges,
        tasks=asked(name, moments, observers),
    )


@clocked("time.shaped")
def shaped():
    graphs = []
    for place, form in enumerate(SHAPED):
        values = [
            varying(form, form, MAKERS, "size", "maker"),
            varying(form, form, SITES, "load", "site"),
        ]
        graphs.append(graph(form, ("hour", "day")[place % 2], values, joined(form, form), MOMENTS, None))
    return dict(schema=stated("clock", ["hour", "day"], "key", False), graphs=graphs)


@clocked("time.steady")
def steady(name, sequenced):
    values = [
        statement("state", {}, dict(key=list(MAKERS), size=[1.0, 2.0, 3.0]), kind="maker"),
        statement("state", {}, dict(key=list(SITES[:3]), load=[4.0, 5.0, 6.0]), kind="site"),
    ]
    edges = [
        statement(
            "state",
            {},
            dict(source=["m0", "m1"], target=["s0", "s1"], present=[True, True], weight=[1.0, 2.0]),
            table="serves",
            ends=("maker", "site"),
        )
    ]
    for place, held in enumerate(values + edges):
        count = len(next(iter(held["rows"].values())))
        held["rows"].update(since=[0] * count, until=[None] * count)
        if sequenced:
            held["rows"]["seq"] = [10 * place + row for row in range(count)]
    return values, edges


@clocked("time.seen")
def seen():
    graphs = []
    values, edges = steady("late", True)
    values.append(
        statement(
            "state",
            {},
            dict(key=["s3"], load=[9.0], since=[4], until=[None], known=[8], seq=[40]),
            kind="site",
        )
    )
    edges.append(
        statement(
            "state",
            {},
            dict(
                source=["m2"],
                target=["s2"],
                present=[True],
                weight=[3.0],
                since=[4],
                until=[None],
                known=[8],
                seq=[41],
            ),
            table="serves",
            ends=("maker", "site"),
        )
    )
    graphs.append(graph("late", "tick", values, edges, MOMENTS, [None] * len(MOMENTS)))
    values, edges = steady("corrected", True)
    values[1]["rows"]["replaced"] = [6, None, None]
    values.append(
        statement(
            "state",
            {},
            dict(key=["s0"], load=[7.0], since=[0], until=[None], known=[6], seq=[40]),
            kind="site",
        )
    )
    graphs.append(graph("corrected", "tick", values, edges, MOMENTS, [None] * len(MOMENTS)))
    values, edges = steady("watched", True)
    values.append(
        statement(
            "state",
            {},
            dict(key=["s3"], load=[8.0], since=[0], until=[None], seq=[40], observers=[["north"]]),
            kind="site",
        )
    )
    edges.append(
        statement(
            "state",
            {},
            dict(
                source=["m2"],
                target=["s2"],
                present=[True],
                weight=[3.0],
                since=[0],
                until=[None],
                seq=[41],
                observers=[["north", "east"]],
            ),
            table="serves",
            ends=("maker", "site"),
        )
    )
    moments = [moment for moment in MOMENTS for _ in range(2)]
    graphs.append(graph("watched", "tick", values, edges, moments, ["north", "south"] * len(MOMENTS)))
    values, edges = steady("several", True)
    values.append(
        statement(
            "state",
            {},
            dict(key=["s0"] * 3, load=[1.0, 2.0, 3.0], since=[2] * 3, until=[6] * 3, seq=[50, 51, 52]),
            kind="site",
        )
    )
    graphs.append(graph("several", "tick", values, edges, MOMENTS, [None] * len(MOMENTS)))
    values, edges = steady("between", True)
    edges.append(
        statement(
            "event",
            {},
            dict(source=["m2"], target=["s2"], present=[True], weight=[3.0], at=[5], seq=[41]),
            table="serves",
            ends=("maker", "site"),
        )
    )
    graphs.append(graph("between", "tick", values, edges, MOMENTS, [None] * len(MOMENTS)))
    return dict(schema=stated("clocks", ["tick"], "sequence", True), graphs=graphs)


@clocked("time.timeless")
def timeless():
    graphs = []
    for name in ("still", "quiet"):
        values, edges = steady(name, False)
        for held in values + edges:
            del held["rows"]["since"], held["rows"]["until"]
            held["form"] = None
        graphs.append(graph(name, None, values, edges, [None] * len(MOMENTS), None))
    return dict(schema=stated("none", [], None, False), graphs=graphs)


@clocked("time.ordered")
def ordered():
    graphs = []
    for name in ("first", "second"):
        values = [
            varying("change", name, MAKERS, "size", "maker"),
            varying("event", name, SITES, "load", "site"),
        ]
        graphs.append(graph(name, None, values, joined("event", name), MOMENTS, None))
    return dict(schema=stated("order", [], "key", False), graphs=graphs)


@clocked("time.spaced")
def spaced():
    values, edges = steady("spaced", False)
    values.append(statement("event", {}, dict(key=["s3", "s3"], load=[7.0, 8.0], at=[5, 6]), kind="site"))
    graphs = [graph("spaced", "tick", values, edges, MOMENTS, None)]
    return dict(schema=stated("clock", ["tick"], "key", False), graphs=graphs)


CASES = dict(forms=shaped, seen=seen, timeless=timeless, ordered=ordered, spaced=spaced)


@pytest.fixture(scope="module")
def cases():
    started = time.perf_counter()
    result = {name: made() for name, made in CASES.items()}
    result["snapshots"] = synthetic.snapshots(result["forms"])
    LOG.info("cases exit cases=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


@pytest.fixture(scope="module")
def banks(cases, tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("time")
    result = {}
    for name, case in cases.items():
        task = synthetic.stage(root, name, case)
        build(task, root / name, 1, ("train",))
        result[name] = root / name
    LOG.info("banks of time exit banks=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


@clocked("time.opened")
def opened(banks, name, **values):
    held = Deriver(banks[name], "cpu", RESIDENT)
    stated = dict(support.several(held.registry), **values)
    return held, configuration(held.registry, dict(view="snapshot"), stated, 5, SPLITS)


@clocked("time.truths")
def truths(held):
    return {graph["name"]: Facts.load(held.truth(graph["index"])) for graph in held.manifest["graphs"]}


@clocked("time.shown")
def shown(held, cfg, name, observer):
    names = held.manifest["observers"]
    for entry in held.viewed(cfg).graphs:
        source = held.manifest["graphs"][entry["source"]]["name"]
        if source == name and entry["observer"] == (-1 if observer is None else names.index(observer)):
            return support.shown(held, entry["index"])
    raise KeyError("the bank shows no graph %s to the observer %s" % (name, observer))


@clocked("time.observed")
def observed(held, graph, step, key):
    dense = graph.dense()
    rows = (dense.obs_step == step) & (dense.obs_entity == graph.keys.index(key))
    values = np.asarray(dense.x_table)[dense.obs_ref[rows]]
    return values[:, held.registry.encoding.layout.index("load")].tolist()


@clocked("time.serving")
def serving(held, graph, step):
    dense = graph.dense()
    rows = (dense.fact_step == step) & dense.fact_present & (dense.fact_code == held.registry.code("serves"))
    return sorted(
        (graph.keys[source], graph.keys[target])
        for source, target in zip(dense.fact_source[rows].tolist(), dense.fact_target[rows].tolist())
    )


@pytest.mark.parametrize("name", ["forms", "snapshots", "seen", "timeless", "ordered", "spaced"])
def test_a_case_of_time_reads_back_from_the_bank_equal_to_what_was_stated(banks, name):
    held, _ = opened(banks, name)
    schema = held.registry.schema
    assert held.manifest["version"] == 8 and schema.tasks["grouping"] is None
    written = support.written(banks[name].parent / (name + ".staging"), schema)
    found = truths(held)
    assert sorted(found) == sorted(written)
    for graph, truth in found.items():
        read = truth.stated(held.registry, held.splits, held.manifest["observers"])
        assert read["graph"] == written[graph]["graph"] and read["nodes"] == written[graph]["nodes"]
        for part in ("observations", "edges"):
            assert support.alike(read[part], written[graph][part]), (graph, part)
        assert support.normal(read["tasks"], schema) == written[graph]["tasks"], graph
        assert read["targets"] == written[graph]["targets"], graph


@pytest.mark.parametrize("place", range(len(SHAPED)))
def test_a_form_is_equal_to_the_same_content_stated_as_snapshots(banks, cases, place):
    form = SHAPED[place]
    stated_in = {held["form"] for held in cases["forms"]["graphs"][place]["values"]}
    assert stated_in == {form}
    if form != "signal":
        assert {held["form"] for held in cases["forms"]["graphs"][place]["edges"]} == {form}
    assert all(
        held["form"] == "snapshot"
        for graph in cases["snapshots"]["graphs"]
        for held in graph["values"] + graph["edges"]
    )
    (left, _), (right, _) = opened(banks, "forms"), opened(banks, "snapshots")
    one, other = truths(left)[form], truths(right)[form]
    assert one.unit == other.unit == ("hour", "day")[place % 2]
    assert sorted(one.arrays) == sorted(other.arrays) and len(one.obs_since) > 0 and len(one.fact_since) > 0
    for name in one.arrays:
        assert np.array_equal(one.arrays[name], other.arrays[name], equal_nan=True), name
    first, second = support.shown(left, place), support.shown(right, place)
    assert first.steps == second.steps == len(MOMENTS)
    for name in first.arrays:
        assert np.array_equal(first.arrays[name], second.arrays[name], equal_nan=True), name


def test_the_forms_reduce_to_the_intervals_written_out():
    rows = pa.table(dict(key=["a", "a", "b"], at=pa.array([3, 7, 3], pa.int64()), size=[1.0, 2.0, 3.0]))
    found = forms.reduced("event", "clock", rows).to_pydict()
    assert found == dict(key=["a", "a", "b"], size=[1.0, 2.0, 3.0], since=[3, 7, 3], until=[4, 8, 4])
    found = forms.reduced("change", "order", rows, keys=["key"]).to_pydict()
    assert found["since"] == [3, 7, 3] and found["until"] == [7, None, None]
    restated = pa.table(
        dict(
            key=["a", "b", "a", "b", "b"],
            at=pa.array([0, 0, 5, 5, 9], pa.int64()),
            size=[1.0, 3.0, 1.0, 4.0, 4.0],
        )
    )
    found = forms.reduced("snapshot", "clock", restated, times=[0, 5, 9]).to_pydict()
    assert sorted(zip(found["key"], found["size"], found["since"], found["until"])) == [
        ("a", 1.0, 0, 9),
        ("b", 3.0, 0, 5),
        ("b", 4.0, 5, None),
    ]
    readings = pa.table(dict(key=["a", "a"], index=pa.array([0, 3], pa.int64()), size=[1.0, 2.0]))
    found = forms.reduced("signal", "clocks", readings, start=10, stride=5).to_pydict()
    assert found["since"] == [10, 25] and found["until"] == [15, 30]
    held = pa.table(dict(key=["a"], since=pa.array([1], pa.int64()), until=pa.array([None], pa.int64())))
    assert forms.reduced("state", "clock", held).to_pydict() == dict(key=["a"], since=[1], until=[None])


def test_a_form_stops_where_it_cannot_state_what_it_is_given():
    readings = pa.table(dict(key=["a"], index=pa.array([0], pa.int64())))
    with pytest.raises(ValueError, match="the form signal counts on a clock"):
        forms.reduced("signal", "order", readings, start=0, stride=1)
    with pytest.raises(ValueError, match="have no time, and the form event states times"):
        forms.reduced("event", "none", readings)
    with pytest.raises(ValueError, match="the form event reads the column at"):
        forms.reduced("event", "clock", readings)
    with pytest.raises(ValueError, match="holds until no later than it holds from"):
        forms.reduced(
            "state",
            "clock",
            pa.table(dict(since=pa.array([4], pa.int64()), until=pa.array([4], pa.int64()))),
        )
    twice = pa.table(dict(key=["a", "a"], at=pa.array([2, 2], pa.int64())))
    with pytest.raises(ValueError, match="two changes of one thing at the time 2"):
        forms.reduced("change", "clock", twice, keys=["key"])
    with pytest.raises(ValueError, match="not among the times of snapshots"):
        forms.reduced("snapshot", "clock", twice, times=[0, 1])
    turned = pa.table(
        dict(
            key=["a", "b", "b", "a"],
            at=pa.array([0, 0, 1, 1], pa.int64()),
            seq=pa.array([0, 1, 0, 1], pa.int32()),
        )
    )
    with pytest.raises(ValueError, match="the snapshot at the time 1 orders two facts against the order"):
        forms.reduced("snapshot", "clock", turned, times=[0, 1])


def test_forms_rules_for_ties_and_choices_of_steps_are_registered_from_outside(monkeypatch):
    for names in (FORMS, TIES):
        monkeypatch.setattr(names, "held", dict(names.held))
    FORMS.register("lasting", forms.Form(lambda table, length: forms.signal(table, 0, length), True))
    rows = pa.table(dict(key=["a"], index=pa.array([2], pa.int64())))
    found = forms.reduced("lasting", "clock", rows, length=3).to_pydict()
    assert found == dict(key=["a"], since=[6], until=[9])
    with pytest.raises(ValueError, match="the form lasting is registered twice"):
        FORMS.register("lasting", forms.Form(forms.state, False))
    with pytest.raises(KeyError, match="no form is registered under the name 'absent'"):
        forms.reduced("absent", "clock", rows)

    class Backwards:
        columns = dict(seq="INT32")

        def keys(self, facts):
            return (-np.asarray(facts["seq"], np.int64),)

    TIES.register("backwards", Backwards())
    with pytest.raises(ValueError, match="the rule for ties backwards is registered twice"):
        TIES.register("backwards", Backwards())
    assert Schema(stated("clock", ["hour"], "backwards", False)).stamps() == dict(
        since="INT64", until="INT64", seq="INT32"
    )
    with pytest.raises(KeyError, match="no rule for ties is registered under the name 'absent'"):
        Schema(stated("clock", ["hour"], "absent", False))
    assert set(("sequence", "key")) <= set(TIES.names()) and SEVERAL.names() == ("first", "last")
    assert STEPS.names()[:2] == ("tasks", "changes") and SEES.names()[:2] == (
        "without_targets",
        "without_scored",
    )
    with pytest.raises(ValueError, match="the choice of steps tasks is registered twice"):
        STEPS.register("tasks", STEPS.named("tasks"))


@pytest.mark.parametrize(
    "model,units,message",
    [
        ("hours", ["hour"], "states its model as one of none, order, clock, clocks, found 'hours'"),
        ("clock", [], "a schema states units of time when its graphs have a clock"),
        ("order", ["hour"], "a schema states units of time when its graphs have a clock"),
        ("clock", ["hour", "hour"], "the units of time of the schema are a list of distinct names"),
    ],
)
def test_a_schema_states_a_model_of_time_with_its_units(model, units, message):
    with pytest.raises(ValueError, match=message):
        Schema(stated(model, units, None, False))


def test_a_schema_states_the_columns_of_time_of_its_model():
    assert Schema(stated("none", [], None, False)).stamps() == {}
    assert Schema(stated("order", [], None, False)).stamps() == dict(since="INT64", until="INT64")
    held = Schema(stated("clocks", ["day"], "sequence", True))
    assert list(held.stamps()) == ["since", "until", "known", "replaced", "seq", "observers"]
    tables = held.tables()
    assert tables["graph"]["required"]["unit"] == "STRING" and tables["task"]["required"]["time"] == "INT64"
    assert tables["task"]["required"]["observer"] == "STRING" and "week" not in tables["task"]["required"]
    assert tables["target"]["required"] == dict(id="INT64", y="DOUBLE", known="INT64")
    timeless = Schema(stated("none", [], None, False)).tables()
    assert "time" not in timeless["task"]["required"] and "known" not in timeless["target"]["required"]
    with pytest.raises(ValueError, match="name observers as true or false"):
        Schema(stated("clock", ["day"], None, "north"))


def test_the_guard_shows_a_fact_where_it_holds_is_known_is_not_replaced_and_may_be_seen():
    since, until = np.asarray([2, 2, 2, 2]), np.asarray([8, 8, 8, 8])
    known, replaced = np.asarray([2, 6, 2, 2]), np.asarray([clock.NEVER, clock.NEVER, 4, clock.NEVER])
    moments = np.asarray([0, 2, 4, 6, 8])
    permitted = visible.allowed(4, [3], [1], 0)
    assert permitted.tolist() == [True, True, True, False]
    begun, ended = visible.visible(since, until, known, replaced, permitted, moments, moments)
    assert list(zip(begun.tolist(), ended.tolist())) == [(1, 4), (3, 4), (1, 2), (1, 1)]
    assert visible.allowed(4, [3], [1], 1).all() and visible.allowed(4, [3], [1], -1).all()
    begun, ended = visible.visible(since, until, known, replaced, permitted, moments, moments + 4)
    assert list(zip(begun.tolist(), ended.tolist())) == [(1, 4), (1, 4), (1, 1), (1, 1)]
    with pytest.raises(ValueError, match="shows no time later than the time of looking"):
        visible.visible(since, until, known, replaced, permitted, moments, moments - 1)
    with pytest.raises(ValueError, match="never decrease"):
        visible.visible(since, until, known, replaced, permitted, moments[::-1], moments[::-1])


def test_a_fact_known_later_than_it_holds_is_invisible_to_a_task_between_the_two_times(banks):
    held, cfg = opened(banks, "seen")
    truth = truths(held)["late"]
    assert truth.fact_known_time.tolist() == [8] and truth.obs_known_time.tolist() == [8]
    assert truth.fact_since[truth.fact_known_row].tolist() == [4]
    graph = shown(held, cfg, "late", None)
    for step, moment in enumerate(MOMENTS):
        assert (("m2", "s2") in serving(held, graph, step)) == (moment >= 8), moment
        assert observed(held, graph, step, "s3") == ([9.0] if moment >= 8 else []), moment
    assert ("m0", "s0") in serving(held, graph, MOMENTS.index(6))


def test_a_corrected_fact_is_seen_in_its_old_value_by_a_task_before_the_correction(banks):
    held, cfg = opened(banks, "seen")
    truth = truths(held)["corrected"]
    assert truth.obs_replaced_time.tolist() == [6] and truth.obs_known_time.tolist() == [6]
    graph = shown(held, cfg, "corrected", None)
    for step, moment in enumerate(MOMENTS):
        assert observed(held, graph, step, "s0") == [4.0 if moment < 6 else 7.0], moment


def test_a_fact_visible_to_one_observer_is_invisible_to_a_task_of_another_at_the_same_time(banks):
    held, cfg = opened(banks, "seen")
    assert held.manifest["observers"] == ["east", "north", "south"]
    truth = truths(held)["watched"]
    assert truth.task_time.tolist() == [moment for moment in MOMENTS for _ in range(2)]
    north, south = shown(held, cfg, "watched", "north"), shown(held, cfg, "watched", "south")
    assert north.steps == south.steps == len(MOMENTS)
    for step in range(len(MOMENTS)):
        assert ("m2", "s2") in serving(held, north, step) and ("m2", "s2") not in serving(held, south, step)
        assert observed(held, north, step, "s3") == [8.0] and observed(held, south, step, "s3") == []
        assert serving(held, south, step) == [("m0", "s0"), ("m1", "s1")]
    rows = np.concatenate((north.label_row, south.label_row))
    assert sorted(rows.tolist()) == sorted(
        place for place, row in enumerate(held.rows) if row["graph"] == "watched"
    )


def test_an_edge_whose_end_an_observer_cannot_see_stops_naming_the_edge_the_end_and_the_observer(banks):
    held, cfg = opened(banks, "seen")
    truth = truths(held)["watched"]
    source, target = np.asarray(truth.fact_source), np.asarray(truth.fact_target)
    edge = int(np.flatnonzero((source == truth.keys.index("m2")) & (target == truth.keys.index("s2")))[0])
    rows, watchers = np.asarray(truth.fact_seen_row), np.asarray(truth.fact_seen_observer)
    ends = np.flatnonzero(np.asarray(truth.obs_entity) == truth.keys.index("s2"))
    seen = np.r_[np.asarray(truth.obs_seen_row), ends]
    watching = np.r_[np.asarray(truth.obs_seen_observer), np.full(len(ends), 1, np.int32)]
    order = np.lexsort((watching, seen))
    changed = truth.replace(
        fact_seen_row=rows[rows != edge],
        fact_seen_observer=watchers[rows != edge],
        obs_seen_row=seen[order],
        obs_seen_observer=watching[order].astype(np.int32),
    )
    assert truth.observers == ["east", "north", "south"]
    found = sequences(changed, held.registry, "tasks", held.looking(cfg)[0])
    south = [sequence for sequence in found if sequence["observer"] == 2]
    with pytest.raises(
        ValueError,
        match="the edge of serves from m2 to s2 of graph watched seen by the observer south holds over the steps "
        "0 to 5, and at one of them its end s2, a node of the kind site, has no observation",
    ):
        view(changed, held.registry, south[0], "last")


def test_three_facts_about_one_entity_at_one_time_are_kept_and_the_view_shows_the_one_chosen(banks):
    held, cfg = opened(banks, "seen")
    truth = truths(held)["several"]
    rows = (truth.obs_entity == truth.keys.index("s0")) & (truth.obs_since == 2)
    assert np.asarray(truth.obs_values)[rows][:, 1].tolist() == [1.0, 2.0, 3.0]
    assert shown(held, cfg, "several", None).steps == len(MOMENTS)
    (sequence,) = sequences(truth, held.registry, "tasks", held.looking(cfg)[0])
    for several, wanted in (("last", 3.0), ("first", 1.0)):
        graph = view(truth, held.registry, sequence, several)
        for step, moment in enumerate(MOMENTS):
            expected = [4.0] if moment < 2 or moment >= 6 else [wanted]
            if several == "first" and 2 <= moment < 6:
                expected = [4.0]
            assert observed(held, graph, step, "s0") == expected, (several, moment)


def test_a_graph_without_time_shows_every_task_the_same_facts(banks):
    held, cfg = opened(banks, "timeless")
    assert held.registry.schema.model == "none" and "time" not in held.rows[0]
    for graph in held.manifest["graphs"]:
        truth = Facts.load(held.truth(graph["index"]))
        assert truth.unit is None and set(truth.obs_since.tolist()) == {clock.ALWAYS}
        assert set(truth.obs_until.tolist()) == {clock.NEVER} and truth.task_time.tolist() == [0] * 6
        shown_graph = support.shown(held, graph["index"])
        assert shown_graph.steps == 1 and shown_graph.period.tolist() == [0]
        assert shown_graph.task_step.tolist() == [0] * 6 and len(shown_graph.label_task) == 6
        assert serving(held, shown_graph, 0) == [("m0", "s0"), ("m1", "s1")]
    spec = cfg["graph"]
    assert cfg["pipeline"][0]["parameters"]["time/periods"] == "steps" and spec["time/steps"] == "tasks"


def test_a_graph_with_order_only_states_positions_and_no_unit(banks):
    held, _ = opened(banks, "ordered")
    assert held.registry.schema.model == "order" and held.registry.schema.units == ()
    for graph in held.manifest["graphs"]:
        truth = Facts.load(held.truth(graph["index"]))
        assert truth.unit is None and graph["unit"] is None
        assert sorted(set(truth.fact_since.tolist())) == list(MOMENTS)
        assert np.array_equal(truth.fact_until, truth.fact_since + 1)
        assert support.shown(held, graph["index"]).steps == len(MOMENTS)


def test_a_bank_holds_graphs_of_different_units_and_states_no_period(banks):
    held, cfg = opened(banks, "forms")
    units = [graph["unit"] for graph in held.manifest["graphs"]]
    assert units == ["hour", "day", "hour", "day", "hour"]
    assert held.registry.schema.units == ("hour", "day")
    names = {parameter.name: parameter for parameter in support.parameters(held.registry, ())}
    assert names["time/periods"].options == ("steps", "every")
    assert all("week" not in row and "period" not in row for row in held.rows)
    assert support.shown(held, 0).period.tolist() == list(range(len(MOMENTS)))
    case = shaped()
    case["graphs"][0]["unit"] = "week"
    with pytest.raises(
        ValueError, match="graph state states the unit of time 'week'; the schema states hour, day"
    ):
        build(
            synthetic.stage(banks["forms"].parent, "strange", case),
            banks["forms"].parent / "odd",
            1,
            ("train",),
        )


def test_facts_about_one_entity_at_one_time_stop_naming_it_where_no_rule_orders_them(banks):
    held, _ = opened(banks, "timeless")
    assert held.registry.schema.ties is None and held.registry.schema.model == "none"
    assert len(truths(held)["still"].fact_source) == 2
    case = timeless()
    rows = case["graphs"][0]["edges"][0]["rows"]
    for name in rows:
        rows[name] = rows[name] + rows[name][:1]
    root = banks["forms"].parent
    with pytest.raises(
        ValueError,
        match="two facts about the edge of serves from m0 to s0 of graph still hold from the time before every "
        "time and the schema states no rule for ties",
    ):
        build(synthetic.stage(root, "untied", case), root / "untied", 1, ("train",))
    case = seen()
    case["graphs"][3]["values"][-1]["rows"]["seq"] = [50, 50, 52]
    with pytest.raises(
        ValueError,
        match="two facts about the node s0 of the kind site of graph several hold from the time 2 and "
        "the rule sequence for ties leaves them unordered",
    ):
        build(synthetic.stage(root, "tied", case), root / "tied", 1, ("train",))


@clocked("time.looked")
def looked(held, cfg, name):
    for entry in held.viewed(cfg).graphs:
        if held.manifest["graphs"][entry["source"]]["name"] == name and entry["observer"] == -1:
            return Graph.load(held.view(entry["index"], cfg))
    raise KeyError("the bank shows no graph %s" % name)


def test_an_event_between_two_steps_is_shown_at_the_first_step_at_or_after_its_time(banks):
    held, cfg = opened(banks, "seen")
    truth = truths(held)["between"]
    found = {}
    for steps in ("tasks", "changes"):
        held, cfg = opened(banks, "seen", **{"time/steps": steps})
        (sequence,) = sequences(truth, held.registry, steps, held.looking(cfg)[0])
        graph = looked(held, cfg, "between")
        found[steps] = [
            int(sequence["moments"][step])
            for step in range(graph.steps)
            if ("m2", "s2") in serving(held, graph, step)
        ]
    assert found == dict(tasks=[6], changes=[5])
    begun, ended = visible.visible([5], [6], [5], [clock.NEVER], [True], [4, 6, 8], [4, 6, 8])
    assert begun.tolist() == [1] and ended.tolist() == [2]
    begun, ended = visible.visible([5], [6], [7], [clock.NEVER], [True], [4, 6, 8], [4, 6, 8])
    assert begun.tolist() == ended.tolist() == [2]


def test_two_events_about_one_node_between_two_task_times_are_shown_by_the_rule_stated_for_several_facts(
    banks,
):
    held = Deriver(banks["spaced"], "cpu", RESIDENT)
    truth = truths(held)["spaced"]
    rows = np.flatnonzero(np.asarray(truth.obs_entity) == truth.keys.index("s3"))
    assert truth.obs_since[rows].tolist() == [5, 6] and truth.obs_until[rows].tolist() == [6, 7]
    assert held.registry.extras["several"] is True
    with pytest.raises(
        ValueError, match=r"the pipeline states no time/several, which has no default; it is one of \['first'"
    ):
        configuration(held.registry, dict(view="snapshot"), {}, 5, SPLITS)
    (sequence,) = sequences(truth, held.registry, "tasks", unstated(held.splits))
    with pytest.raises(ValueError, match="facts about one entity that a view can show at one step"):
        view(truth, held.registry, sequence, None)
    for several, wanted in (("first", 7.0), ("last", 8.0)):
        graph = view(truth, held.registry, sequence, several)
        for step, moment in enumerate(MOMENTS):
            assert observed(held, graph, step, "s3") == ([wanted] if moment == 6 else []), (several, moment)
    held, cfg = opened(banks, "spaced", **{"time/several": "first"})
    assert observed(held, looked(held, cfg, "spaced"), MOMENTS.index(6), "s3") == [7.0]


def broken(starts, stops):
    raise AssertionError("a rule registered first is not the rule of validation")


class Broken:
    def serves(self, registry):
        return True

    def __call__(self, facts, registry):
        raise AssertionError("a choice of steps registered first is not the choice of validation")


def test_the_rule_for_several_facts_is_stated_where_a_bank_holds_them_and_validation_names_its_own(
    banks, monkeypatch
):
    from graphspec.engine.transform import validate

    seen, forms_bank = Deriver(banks["seen"], "cpu", RESIDENT), Deriver(banks["forms"], "cpu", RESIDENT)
    assert seen.registry.extras["several"] is True and forms_bank.registry.extras["several"] is False
    with pytest.raises(
        ValueError, match=r"the pipeline states no time/several, which has no default; it is one of \['first'"
    ):
        configuration(seen.registry, dict(view="snapshot"), {}, 5, SPLITS)
    assert "time/several" not in {parameter.name for parameter in specs.settings(forms_bank.registry, ())}
    truth = truths(seen)["several"]
    before = validate(truth, seen.registry, seen.splits)
    monkeypatch.setattr(SEVERAL, "held", dict(broken=broken, **SEVERAL.held))
    monkeypatch.setattr(STEPS, "held", dict(broken=Broken(), **STEPS.held))
    after = validate(truth, seen.registry, seen.splits)
    assert len(before) == len(after) > 0
    for left, right in zip(before, after):
        assert all(np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True) for name in left.arrays)
    with pytest.raises(
        ValueError, match="facts about one entity that a view can show at one step, and no time/several"
    ):
        view(truth, seen.registry, sequences(truth, seen.registry, "tasks", unstated(seen.splits))[0], None)


def test_periods_are_a_view_and_need_not_increase(banks):
    held, cfg = opened(banks, "ordered")
    graph = support.shown(held, 0)
    turned = graph.replace(period=np.asarray([3, 3, 1, 1, 2, 2], np.int32))
    from graphspec.views.graph import verify

    verify(turned, held.registry)
    dense = turned.dense()
    assert dense.period.tolist() == [3, 3, 1, 1, 2, 2]
    from graphspec.budgets import GraphLimits
    from graphspec.engine import segments

    every = support.steps(held.registry, {"time/periods": "every", "time/period_length": 4})[0]
    derived = segments.whole(dense, support.fitted(held), every, GraphLimits("cpu"), None)
    assert derived.period.tolist() == [0, 0, 0, 0, 1, 1]
    kept = support.steps(held.registry, {})[0]
    assert segments.whole(dense, support.fitted(held), kept, GraphLimits("cpu"), None).period.tolist() == [
        3,
        3,
        1,
        1,
        2,
        2,
    ]


@pytest.fixture
def prepared(banks, monkeypatch, request):
    started = time.perf_counter()
    cfg = inline(banks[request.param], ("snapshot",), monkeypatch)["snapshot"]
    held = Deriver(banks[request.param], "cpu", RESIDENT)
    blocks = held.blocks(0, 4, cfg)
    LOG.info("prepared exit case=%s seconds=%.3f", request.param, time.perf_counter() - started)
    return banks[request.param], held, cfg, blocks


@pytest.mark.parametrize("prepared", ["forms", "seen", "timeless", "ordered"], indirect=True)
def test_a_case_of_time_completes_one_fit_on_cpu(prepared, tmp_path, statement):
    bank, held, cfg, blocks = prepared
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], state(statement))
    LOG.info("fit of time exit bank=%s seconds=%.3f", bank.name, time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(held.members(cfg, 0))
    assert np.isfinite(saved["raw_val_mse"]) and saved["holdout_rows"] == len(held.members(cfg, 1))


def test_a_database_gives_the_facts_of_two_clocks_and_their_observers_as_the_staging_does(banks, tmp_path):
    pytest.importorskip("ladybug")
    from graphspec.facts.stored import STORED
    from graphspec.infra.files import write_json
    from graphspec.sources.ladybug import export

    root = banks["seen"].parent
    export(root / "seen.staging", tmp_path / "seen.lbug")
    schema = read_json(root / "seen.task.json")["schema"]
    write_json(tmp_path / "task.json", dict(name="seen", database="seen.lbug", schema=schema))
    build(tmp_path / "task.json", tmp_path / "bank", 1, ("train",))
    assert read_json(tmp_path / "bank" / "manifest.json")["observers"] == ["east", "north", "south"]
    for index in range(4):
        left = Facts.load(banks["seen"] / "truth" / ("graph_%04d" % index))
        right = Facts.load(tmp_path / "bank" / "truth" / ("graph_%04d" % index))
        assert left.unit == right.unit == "tick"
        for name in STORED:
            assert np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True), name


def test_the_steps_of_a_view_may_be_the_times_of_change(tmp_path):
    from graphspec import GraphTables, Stage, write_task

    mapping = dict(stated("clock", ["day"], "key", False), tasks=support.asked(["maker", "site"]))
    del mapping["edges"]["beside"], mapping["columns"]["ask"]
    schema = Schema(mapping)
    graph = GraphTables(0, "plain", "day", schema)
    graph.nodes("maker", ["a"]).nodes("site", ["b"])
    graph.observe("maker", ["a", "a"], dict(since=[3, 10], until=[10, None]), size=[1.0, 2.0])
    graph.observe("site", ["b"], dict(since=3, until=None), load=[5.0])
    graph.edges("serves", "maker", "site", ["a"], ["b"], dict(since=5, until=20), present=True, weight=[1.5])
    key = 0
    for moment, values in ((10, [0.0, 1.0]), (20, [2.0, 3.0])):
        for kind, node, value, split in (
            ("maker", "a", values[0], "train"),
            ("site", "b", values[1], "validation"),
        ):
            graph.tasks(
                "outcome", [key], moment, dict(holder=(kind, [node])), [split], [value], [None], None, None
            )
            key += 1
    with Stage(tmp_path / "source") as stage:
        graph.write(stage)
    task = write_task(tmp_path / "task.json", "plain", tmp_path / "source", schema)
    manifest = build(task, tmp_path / "bank", 1, ("train",))
    assert manifest["spec"]["extras"]["target"] == "number" and manifest["graphs"][0]["tasks"] == 4
    held = Deriver(tmp_path / "bank", "cpu", RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), {"time/steps": "changes"}, 5, SPLITS)
    assert cfg["pipeline"][0] == dict(operation="graphspec.periods", parameters={"time/periods": "steps"})
    truth = Facts.load(held.truth(0))
    (sequence,) = sequences(truth, held.registry, "changes", held.looking(cfg)[0])
    assert sequence["moments"].tolist() == [3, 5, 10, 20] and sequence["observer"] == -1
    assert sequence["task_step"].tolist() == [2, 2, 3, 3]
    shown_graph = Graph.load(held.view(0, cfg))
    dense = shown_graph.dense()
    assert shown_graph.steps == 4 and dense.fact_step.tolist() == [1, 2]
    assert dense.obs_step.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.asarray(dense.x_table)[dense.obs_ref[dense.obs_entity == 0]][:, 0].tolist() == [1, 1, 2, 2]
    assert shown_graph.task_step.tolist() == [2, 2, 3, 3] and shown_graph.label_task.tolist() == [0, 1, 2, 3]
    assert [row["time"] for row in held.rows] == [10, 10, 20, 20]
    (sequence,) = sequences(truth, held.registry, "tasks", held.looking(cfg)[0])
    assert sequence["moments"].tolist() == [10, 20] and sequence["task_step"].tolist() == [0, 0, 1, 1]


def test_a_choice_of_steps_registered_from_outside_is_offered_where_it_serves(banks, monkeypatch):
    from graphspec.views.view import offered
    from graphspec.views.visible import unstated

    monkeypatch.setattr(STEPS, "held", dict(STEPS.held))

    class Halved:
        def serves(self, registry):
            return registry.name == "ordered"

        def __call__(self, facts, registry):
            (whole,) = STEPS.named("tasks")(facts, registry)
            kept = np.arange(0, len(whole["moments"]), 2)
            placed = np.isin(whole["task_step"], kept)
            return [
                dict(
                    whole,
                    moments=whole["moments"][kept],
                    looking=whole["looking"][kept],
                    tasks=whole["tasks"][placed],
                    task_step=(whole["task_step"][placed] // 2).astype(np.int32),
                )
            ]

    STEPS.register("halved", Halved())
    held, _ = opened(banks, "ordered")
    other, _ = opened(banks, "timeless")
    assert offered(held.registry) == ("tasks", "changes", "halved")
    assert offered(other.registry) == ("tasks", "changes")
    truth = Facts.load(held.truth(0))
    with pytest.raises(
        ValueError, match="the choice of steps halved does not give every task of graph first that holds"
    ):
        sequences(truth, held.registry, "halved", unstated(held.splits))
    (chosen,) = STEPS.named("halved")(truth, held.registry)
    chosen.update(split=-1, rules=(), trained=(), scored=(), labels=np.zeros(0, np.int64), matures=())
    assert view(truth, held.registry, chosen, "last").steps == 3
    assert held.registry.extras["several"] is False
    with pytest.raises(
        ValueError,
        match="the view of graph first shows at the time 4 several facts about the node s0, and no time/several",
    ):
        view(truth, held.registry, chosen, None)
    names = {parameter.name: parameter for parameter in specs.settings(held.registry, ())}
    assert names["time/steps"].options == ("tasks", "changes", "halved")
    assert names["time/steps"].default == "tasks"


def test_a_case_is_not_changed_by_being_restated(cases):
    before = copy.deepcopy(cases["forms"])
    synthetic.snapshots(cases["forms"])
    assert cases["forms"] == before
