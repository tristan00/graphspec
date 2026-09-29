import copy
import logging
import sys
import time

import fixture
import numpy as np
import pyarrow as pa
import pytest
import support

from graphspec import ENCODERS, TYPES, inspect, join, synthetic
from graphspec import spec as specs
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.schema import VALUE, Encoding, Schema, fitted
from graphspec.truth import VERSION, Facts, conforms, distinct
from tuning.build import build
from tuning.data import Deriver
from tuning.statement import state
from tuning.training import fit

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("schema")


def lacking(mapping, *path):
    held = copy.deepcopy(mapping)
    part = held
    for name in path[:-1]:
        part = part[name]
    del part[path[-1]]
    return held


def test_the_schema_is_built_from_a_mapping_and_written_back_to_one():
    mapping = synthetic.varied("task")["schema"]
    schema = Schema(mapping)
    assert schema.mapping() == mapping and Schema(schema.mapping()).fingerprint() == schema.fingerprint()
    assert schema.layout() == ("size", "speed", "colour", "open")
    assert schema.edges["near"]["directed"] is False and schema.edges["links"]["column"] == "predicate"
    assert schema.model == "clock" and schema.units == ("step",) and schema.graph == ("season",)
    assert schema.ties == "sequence" and schema.observers is False
    assert schema.events() == ("buy", "move", "sell") and schema.series() == ("outcome",)
    assert schema.aim() == "number" and schema.asked_columns() == ("size", "speed")
    assert Schema(synthetic.varied("path")["schema"]).series() == ("level", "trend")
    held = Schema(synthetic.varied("node")["schema"])
    assert held.aimed_kinds() == ("outcome",) and held.tasks["signals"] == 0
    assert held.tasks["kinds"]["outcome"]["references"] == ("holder",)


@pytest.mark.parametrize(
    "path,message",
    [
        (("version",), "the schema lacks version"),
        (("time",), "the schema lacks time"),
        (("time", "units"), "the time of the schema lacks units"),
        (("time", "model"), "the time of the schema lacks model"),
        (("time", "ties"), "the time of the schema lacks ties"),
        (("observers",), "the schema lacks observers"),
        (("columns", "size", "type"), "the column size lacks type"),
        (("columns", "size", "encoder"), "the column size lacks encoder"),
        (("columns", "mode"), "the columns of the edge table visits have no stated type: mode"),
        (("columns", "season"), "the columns of the graph have no stated type: season"),
        (("graph",), "the schema lacks graph"),
        (("nodes", "place", "columns"), "the kind of node place lacks columns"),
        (("edges", "visits", "endpoints"), "the edge table visits lacks endpoints"),
        (("edges", "near", "directed"), "the edge table near lacks directed"),
        (("edges", "links", "relation"), "the edge table links lacks relation"),
        (("edges", "links", "relation", "column"), "the relation of the edge table links lacks column"),
        (("tasks",), "the schema lacks tasks"),
        (("tasks", "signals"), "the tasks lacks signals"),
        (("tasks", "grouping"), "the tasks lacks grouping"),
        (("tasks", "key"), "the tasks lacks key"),
        (("tasks", "references"), "the tasks lacks references"),
        (("tasks", "kinds"), "the tasks lacks kinds"),
        (("tasks", "references", "place", "sort"), "the reference place lacks sort"),
        (("tasks", "references", "place", "kinds"), "the reference place lacks kinds"),
        (("tasks", "references", "team"), "the kind of task buy names references not stated: team"),
        (("tasks", "kinds", "buy", "references"), "the kind of task buy lacks references"),
        (("tasks", "kinds", "buy", "columns"), "the kind of task buy lacks columns"),
        (("tasks", "kinds", "buy", "target"), "the kind of task buy lacks target"),
        (("tasks", "kinds", "buy", "target", "series"), "the target of the kind of task buy lacks series"),
        (("tasks", "kinds", "buy", "weight"), "the kind of task buy lacks weight"),
        (("tasks", "kinds", "buy", "absent"), "the kind of task buy lacks absent"),
        (("tasks", "kinds", "buy", "event"), "the kind of task buy lacks event"),
        (("tasks", "kinds", "buy", "matures"), "the kind of task buy lacks matures"),
    ],
)
def test_a_schema_with_a_statement_absent_stops_naming_it(path, message):
    with pytest.raises(ValueError, match=message):
        Schema(lacking(synthetic.varied("task")["schema"], *path))


def test_a_schema_of_another_version_or_with_an_unknown_name_is_refused():
    mapping = synthetic.varied("task")["schema"]
    with pytest.raises(ValueError, match="the schema has version 0; this code reads version 5"):
        Schema(dict(mapping, version=0))
    with pytest.raises(KeyError, match="no column type is registered under the name 'colourful'"):
        Schema(dict(mapping, columns=dict(mapping["columns"], size=dict(type="colourful", encoder="value"))))
    with pytest.raises(KeyError, match="no encoder is registered under the name 'absent'"):
        Schema(dict(mapping, columns=dict(mapping["columns"], size=dict(type="number", encoder="absent"))))
    with pytest.raises(ValueError, match="the type category, which the encoder standard does not read"):
        Schema(
            dict(mapping, columns=dict(mapping["columns"], mode=dict(type="category", encoder="standard")))
        )
    with pytest.raises(ValueError, match="joins kinds of node not stated: planet"):
        Schema(dict(mapping, edges=dict(mapping["edges"], orbit=synthetic.edge("planet", "place", []))))
    with pytest.raises(ValueError, match="holds fields it does not know: colour"):
        Schema(dict(mapping, colour="red"))


class Count:
    accepts = ("INT32",)
    written = pa.int32()
    symbols = False
    ordered = True

    def read(self, column):
        return np.asarray(column.to_numpy(zero_copy_only=False), np.float32)

    def missing(self, stated):
        return np.isnan(stated)

    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else int(value) for value in codes]


class Halved:
    types = ("count",)

    def fit(self, codes, dictionary, column):
        return dict(most=float(np.max(codes)))

    def widths(self, state):
        return 2, 0

    def encode(self, codes, state, dictionary):
        held = np.nan_to_num(np.asarray(codes, np.float32))
        return np.stack((held / 2, held / state["most"]), 1), np.zeros((len(held), 0), np.float32)


def test_column_types_and_encoders_are_registered_by_name_from_outside(monkeypatch):
    for names in (TYPES, ENCODERS):
        monkeypatch.setattr(names, "held", dict(names.held))
    TYPES.register("count", Count())
    ENCODERS.register("halved", Halved())
    with pytest.raises(ValueError, match="the column type count is registered twice"):
        TYPES.register("count", Count())
    with pytest.raises(ValueError, match="the encoder halved is registered twice"):
        ENCODERS.register("halved", Halved())
    schema = support.stated(
        dict(shelf=["stock", "value"], crate=["stock"]),
        {},
        dict(stock=dict(type="count", encoder="halved"), value=dict(type="number", encoder="value")),
        support.asked(["shelf"]),
    )
    table = pa.table(
        {
            "since": pa.array([0, 1, 2], pa.int64()),
            VALUE + "stock": pa.array([4, None, 10], pa.int32()),
            VALUE + "value": pa.array([1.0, 2.0, 3.0], pa.float32()),
        }
    )
    tables = dict(
        graph=pa.table(dict(name=["one"])),
        observations=dict(shelf=table, crate=table.select(["since", VALUE + "stock"])),
        edges={},
    )
    assert distinct(schema, tables) == {}
    shown = dict(stock=[np.asarray([4.0], np.float32)], value=[np.asarray([1.0, 2.0], np.float32)])
    encoding = fitted(Encoding(schema, {}, None, {}), shown)
    assert encoding.states["stock"] == dict(most=4.0)
    assert encoding.features == 4 and encoding.values == 3
    assert encoding.width(schema.nodes["shelf"]) == 4 and encoding.width(schema.nodes["crate"]) == 2
    encoded = encoding.nodes(np.asarray([[10, np.nan], [4, 3]], np.float32))
    assert encoded.tolist() == [[5.0, 2.5, 0.0, 0.0], [2.0, 1.0, 3.0, 1.0]]


def test_a_bank_keeps_every_value_stated_in_its_dictionaries_and_encoders_fit_on_what_they_are_shown():
    schema = support.stated(
        dict(shelf=["shade", "height"]),
        {},
        dict(shade=dict(type="category", encoder="onehot"), height=dict(type="number", encoder="standard")),
        support.asked(["shelf"]),
    )
    table = pa.table(
        {
            "since": pa.array([0, 1, 2, 3, 4], pa.int64()),
            VALUE + "shade": pa.array(["dark", "light", None, "pale", "dark"], pa.string()),
            VALUE + "height": pa.array([1.0, 3.0, None, 100.0, 200.0], pa.float32()),
        }
    )
    tables = dict(graph=pa.table(dict(name=["one"])), observations=dict(shelf=table), edges={})
    assert distinct(schema, tables) == dict(shade={"dark", "light", "pale"})
    stated = Encoding(schema, dict(shade=["dark", "light", "pale"]), None, {})
    with pytest.raises(ValueError, match="features needs the fitted encoders"):
        stated.features
    shown = dict(shade=[np.asarray([0.0, 1.0], np.float32)], height=[np.asarray([1.0, 3.0], np.float32)])
    encoding = fitted(stated, shown)
    assert encoding.dictionaries == dict(shade=["dark", "light", "pale"])
    assert encoding.states["shade"] == dict(values=["dark", "light"])
    assert encoding.states["height"] == dict(mean=2.0, sd=pytest.approx(2.0**0.5))
    encoded = encoding.nodes(np.asarray([[0, 1], [2, 3], [np.nan, np.nan]], np.float32))
    assert encoded[:, :3].tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    assert encoded[:, 3] == pytest.approx([-(0.5**0.5), 0.5**0.5, 0.0])
    assert encoded[:, 4:].tolist() == [[1, 1], [1, 1], [0, 0]]
    again = Encoding(schema, **copy.deepcopy(encoding.declaration()))
    assert np.array_equal(again.nodes(np.asarray([[1, 5]], np.float32)), encoding.nodes([[1, 5]]))
    blind = fitted(stated, dict(shown, shade=[]))
    assert blind.states["shade"] == dict(values=[])
    assert blind.nodes(np.asarray([[2, 3]], np.float32))[0, :1].tolist() == [1.0]
    with pytest.raises(
        ValueError,
        match="the column height has fewer than two differing values in what the views of the splits trained on "
        "show, and cannot be standardised",
    ):
        fitted(stated, dict(shown, height=[np.asarray([1.0], np.float32)]))


@pytest.mark.parametrize("unseen", ["stop", "shared"])
def test_the_onehot_encoder_states_what_it_does_with_a_value_unseen_in_training(unseen):
    from graphspec.schema import OneHot

    encoder, dictionary = OneHot(unseen), ["low", "mid", "high"]
    state = encoder.fit(np.asarray([0, 2]), dictionary, "shade")
    assert state == dict(values=["low", "high"])
    values, flags = encoder.encode(np.asarray([0, 2, np.nan], np.float32), state, dictionary)
    assert values.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0]] and flags.tolist() == [[1], [1], [0]]
    wider = ["low", "lower", "mid", "high"]
    assert encoder.encode(np.asarray([0, 3], np.float32), state, wider)[0].tolist() == [[1, 0, 0], [0, 1, 0]]
    if unseen == "stop":
        with pytest.raises(ValueError, match="stops on a value unseen in training and met the values mid"):
            encoder.encode(np.asarray([0, 1], np.float32), state, dictionary)
    else:
        found = encoder.encode(np.asarray([0, 1], np.float32), state, dictionary)[0]
        assert found.tolist() == [[1, 0, 0], [0, 0, 1]]
    assert ENCODERS.named("onehot").unseen == "shared"
    with pytest.raises(TypeError, match="unseen"):
        OneHot()
    with pytest.raises(ValueError, match="one of stop, shared, found 'ignore'"):
        OneHot("ignore")


def test_the_varied_graph_builds_with_what_the_check_names(varied):
    manifest = read_json(varied / "manifest.json")
    declared = manifest["spec"]
    assert manifest["version"] == VERSION == 8 and manifest["trained"] == ["train"]
    assert declared["schema"] == synthetic.varied("task")["schema"]
    assert manifest["splits"] == ["train", "validation"] and declared["extras"]["single"] is True
    assert declared["extras"]["events"] is True and declared["extras"]["target"] == "number"
    assert declared["encoding"]["states"] is None and "widths" not in declared
    edges = {edge["name"]: edge for edge in declared["edges"]}
    assert edges["visits"]["attributes"] == ["weight", "mode"]
    assert declared["schema"]["columns"]["mode"]["type"] == "category"
    assert edges["near"]["symmetric"] and edges["near"]["reverse"] == -1 and not edges["visits"]["symmetric"]
    taken = ["lends", "likes", "seeks", "sells", "sends", "shows"]
    assert declared["encoding"]["relations"]["links"] == taken
    assert all(edges[name]["origin"] == ["base", "links"] for name in taken)
    assert declared["extras"]["base"] == ["visits", "owns", "member", "near", "located"] + taken
    opened = Deriver(varied, "cpu", RESIDENT)
    registry = support.fitted(opened)
    assert registry.widths == dict(person=11, place=2, item=7, team=2) and registry.features == 11
    assert registry.attribute_width("visits") == 2 + len(registry.encoding.states["mode"]["values"]) + 2
    for graph in manifest["graphs"]:
        truth = Facts.load(opened.truth(graph["index"]))
        assert "obs_x" not in truth.arrays and truth.obs_values.shape[1] == 4 and truth.unit == "step"
        place = np.asarray(truth.kinds)[truth.obs_entity] == opened.registry.kind("place")
        assert np.isnan(np.asarray(truth.obs_values)[place][:, 1:]).all()
        near = np.asarray(truth.fact_code) == opened.registry.code("near")
        assert near.any() and np.all(truth.fact_source[near] <= truth.fact_target[near])
        found = set(np.unique(truth.fact_code).tolist())
        assert found & {opened.registry.code(name) for name in taken}


def test_the_varied_graph_reads_back_from_the_bank_equal_to_what_was_stated(varied):
    opened = Deriver(varied, "cpu", RESIDENT)
    assert set(opened.rows[0]) == {"graph", "kind", "time", "ticket", "week", "y"}
    assert opened.split["series"][0]["name"] == synthetic.OUTCOME and len(opened.split["series"]) == 1
    assert sorted(opened.split["indices"]) == ["train", "validation"]
    schema = opened.registry.schema
    stated = support.written(varied.parent / "varied.staging", schema)
    for graph in opened.manifest["graphs"]:
        truth = Facts.load(opened.truth(graph["index"]))
        found = truth.stated(opened.registry, opened.splits, opened.manifest["observers"])
        wanted = stated[truth.name]
        assert found["graph"] == wanted["graph"] and found["nodes"] == wanted["nodes"]
        for part in ("observations", "edges"):
            assert support.alike(found[part], wanted[part]), (truth.name, part)
        assert support.normal(found["tasks"], schema) == wanted["tasks"]
        assert found["targets"] == wanted["targets"]
        assert any(value is None for value in found["observations"]["person"][VALUE + "colour"])
        assert set(found["edges"]["visits"][VALUE + "mode"]) <= {"foot", "road", "rail", "air", None}


def test_the_varied_graph_completes_one_fit_on_cpu(varied, varied_packed, tmp_path, statement):
    opened, cfg = Deriver(varied, "cpu", RESIDENT), varied_packed["snapshot"]
    assert cfg["model"]["snapshot/spatial"] == "relational"
    blocks = opened.blocks(0, 4, cfg)
    started = time.perf_counter()
    fit(varied, tmp_path, cfg, blocks, 16, started + statement["seconds"], state(statement))
    LOG.info("varied fit exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(opened.members(cfg, 0))
    assert np.isfinite(saved["raw_val_mse"]) and saved["holdout_rows"] == len(opened.members(cfg, 1))


def test_a_schema_is_checked_after_an_operation_for_what_the_operation_changes(varied):
    opened = Deriver(varied, "cpu", RESIDENT)
    registry = support.fitted(opened)
    dense = support.shown(opened, 0).dense()
    conforms(dense, registry, ("time", "observations", "facts"))
    moved = np.asarray(dense.fact_source).copy()
    moved[np.flatnonzero(dense.fact_code == registry.code("visits"))[0]] = dense.fact_target[
        np.flatnonzero(dense.fact_code == registry.code("visits"))[0]
    ]
    broken = dense.replace(fact_source=moved)
    conforms(broken, registry, ("time", "observations"))
    with pytest.raises(ValueError, match="a fact of visits between kinds of node the relation does not join"):
        conforms(broken, registry, ("facts",))
    with pytest.raises(ValueError, match="changes the schema does not know: colour"):
        conforms(dense, registry, ("colour",))


def test_inspect_prints_the_schema_of_a_task_and_of_a_bank(varied, capsys, monkeypatch):
    from tuning.__main__ import main

    stated = inspect(varied.parent / "varied.task.json")
    assert "time: one clock, units step, ties by sequence" in stated and "observers: none" in stated
    assert "graph: season category by onehot" in stated
    assert "  place: size number by standard" in stated and ", width" not in stated
    assert "  near: place to place, without direction, relation declared, no columns" in stated
    assert "relation taken from the column predicate, mode category by onehot" in stated
    with pytest.raises(TypeError, match="encoding"):
        Schema(read_json(varied.parent / "varied.task.json")["schema"]).text()
    monkeypatch.setattr(sys, "argv", ["graphspec", "inspect", str(varied)])
    main()
    printed = capsys.readouterr().out
    assert "schema version 5" in printed and "  place: size number by standard\n" in printed
    assert ", width" not in printed and "  lends: from links" in printed
    assert "  reference person: node of person" in printed and "keyed by ticket, grouped by week" in printed
    assert (
        "  kind buy: references person, place, team; size number by standard, speed number by standard; "
        "target number of outcome; not weighted; absent stop; "
        "seen as an event between person and place; nothing matures" in printed
    )


def test_a_bank_of_version_7_is_refused_with_an_error_naming_its_version(ground):
    old = ground / "01_plain" / "bank"
    with pytest.raises(ValueError, match="has version 7; this code reads version 8"):
        Deriver(old, "cpu", RESIDENT)
    with pytest.raises(ValueError, match="has version 7; this code reads version 8"):
        inspect(old)


@pytest.fixture(scope="module")
def built(ground, tmp_path_factory):
    started = time.perf_counter()
    found = {}
    for entry in fixture.configurations(ground)[0]:
        source = entry["source"]
        if source["name"] not in found:
            root = tmp_path_factory.mktemp("ground-" + source["name"])
            shape, settings = fixture.shaped(source)
            task = synthetic.generate(root, "source", source["graphs"], source["seed"], shape, **settings)
            build(task, root / "bank", 1, ("train",))
            found[source["name"]] = root
    LOG.info("built exit sources=%d seconds=%.3f", len(found), time.perf_counter() - started)
    return found


@pytest.mark.parametrize("place", range(10))
def test_the_inputs_of_a_configuration_of_the_fixture_are_equal_under_the_names_of_version_8(
    ground, built, place
):
    entry = fixture.configurations(ground)[0][place]
    root = built[entry["source"]["name"]]
    started = time.perf_counter()
    schema, level = Schema(fixture.shaped(entry["source"])[0]["schema"]), entry["source"]["level"]
    old = fixture.tables(ground / entry["folder"] / entry["files"]["staged"], schema, level)
    new = support.staged(root / "source.staging")
    assert sorted(map(str, old)) == sorted(map(str, new))
    for key, table in old.items():
        assert support.sorted_by_key(table).equals(support.sorted_by_key(new[key])), key
    held = fixture.arrays(ground / entry["folder"] / entry["files"]["input"], level)
    opened = Deriver(root / "bank", "cpu", RESIDENT)
    assert [graph["name"] for graph in opened.manifest["graphs"]] == entry["graphs"]
    for graph in opened.manifest["graphs"]:
        dense = support.shown(opened, graph["index"]).dense()
        assert fixture.differing(dense.arrays, held[dense.name]) == []
    assert read_json(root / "bank" / "supervision.json") == fixture.supervision(
        ground / entry["folder"] / entry["files"]["bank"], schema, level, old
    )
    LOG.info("fixture inputs exit name=%s seconds=%.3f", entry["name"], time.perf_counter() - started)


@pytest.mark.parametrize("place", range(2))
def test_the_joined_input_of_the_fixture_is_equal_under_the_names_of_version_8(ground, built, place):
    entries, joined = fixture.configurations(ground)
    entry = joined[place]
    source = {held["name"]: held for held in entries}[entry["configuration"]]["source"]
    opened = Deriver(built[source["name"]] / "bank", "cpu", RESIDENT)
    started = time.perf_counter()
    spec = specs.configured(specs.space(opened.registry, tuple(entry["groups"])), entry["values"])
    assert spec == fixture.spec(entry["spec"])
    members = [support.shown(opened, graph["index"]).dense() for graph in opened.manifest["graphs"]]
    held = fixture.joined(ground / entry["folder"] / entry["files"]["input"], source["level"], members)
    assert fixture.differing(join.joined(members, support.fitted(opened), spec).arrays, held) == []
    LOG.info("fixture joined exit name=%s seconds=%.3f", entry["name"], time.perf_counter() - started)
