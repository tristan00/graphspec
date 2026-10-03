import copy
import importlib
import logging
import sys
import time
from pathlib import Path

import fixture
import keyed
import numpy as np
import pytest
import support
from registered import everything, parities, parity, running, unchecked, uncounted, unlooking

from conftest import SPLITS, Inline
from graphspec import (
    ENCODERS,
    NEGATIVES,
    OPERATIONS,
    REPRESENTATIONS,
    SEES,
    SEVERAL,
    Operation,
    Phase,
    Representation,
)
from graphspec.budgets import GraphLimits
from graphspec.engine import segments, transform
from graphspec.engine.segments import Derivation
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.negatives import Rule
from graphspec.views.graph import Dense
from tuning.bank.bank import Deriver
from tuning.fits.training import fit
from tuning.searches.selector import random_suggest
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("fixture")
GONE = dict(
    first="c390954",
    second="c390954",
    starts="c390954",
    affinity="c390954",
    global_affinity="c390954",
    period_first_keys="c390954",
    period_first_rows="c390954",
    period_kind="c390954",
    spatial_order="8458102",
    spatial_code_offsets="8458102",
)
ADDED = dict(
    row_kind="c390954",
    row_left="follow-up of links and splits",
    member_place="c390954",
    affinity_keys="c390954",
    affinity_counts="c390954",
    affinity_steps="c390954",
    member_row="9249781",
    member_name="9249781",
    member_entity="9249781",
    member_observed="9249781",
    whole_row="9249781",
    whole_name="9249781",
    row_task="9249781",
    event_task="9249781",
    event_step="9249781",
    event_identity="follow-up of 2.8",
    step_time="follow-up of 2.8",
    step_token="follow-up of 2.8",
)
CHANGED = dict(events="9249781", key="9249781", task="9249781", walks="c390954", period_walks="c390954")
PIPELINED = {
    ("02_periods_sample_coupling", "stage_3_nodes"): "e6f9e7752a4d6e9d",
    ("02_periods_sample_coupling", "stage_4_derived_relations"): "e6f9e7752a4d6e9d",
    ("02_periods_sample_coupling", "stage_5_relations"): "e6f9e7752a4d6e9d",
    ("02_periods_sample_coupling", "stage_6_edge_filters"): "e6f9e7752a4d6e9d",
    ("02_periods_sample_coupling", "stage_7_relation_merging"): "e6f9e7752a4d6e9d",
    ("02_periods_sample_coupling", "representation"): "38c6360a41ae17eb",
    ("05_sample_drop_neighbors", "stage_3_nodes"): "48d24f17b5931ed2",
    ("05_sample_drop_neighbors", "stage_4_derived_relations"): "48d24f17b5931ed2",
    ("05_sample_drop_neighbors", "stage_5_relations"): "48d24f17b5931ed2",
    ("05_sample_drop_neighbors", "stage_6_edge_filters"): "679e314692102003",
    ("05_sample_drop_neighbors", "stage_7_relation_merging"): "679e314692102003",
    ("05_sample_drop_neighbors", "representation"): "96e3bca1f6206874",
    ("06_project_complete_walks", "stage_4_derived_relations"): "a0aa6fadad7ea7a7",
    ("06_project_complete_walks", "stage_5_relations"): "a0aa6fadad7ea7a7",
    ("06_project_complete_walks", "stage_6_edge_filters"): "5584f7d1fae02c4d",
    ("06_project_complete_walks", "stage_7_relation_merging"): "5584f7d1fae02c4d",
    ("06_project_complete_walks", "representation"): "d10c2dce2e2f3ad4",
    ("07_periods_hops_ppr_coupling", "stage_4_derived_relations"): "252100c0f13eb1a4",
    ("07_periods_hops_ppr_coupling", "stage_5_relations"): "252100c0f13eb1a4",
    ("07_periods_hops_ppr_coupling", "stage_6_edge_filters"): "252100c0f13eb1a4",
    ("07_periods_hops_ppr_coupling", "stage_7_relation_merging"): "252100c0f13eb1a4",
    ("07_periods_hops_ppr_coupling", "representation"): "72edae7e0410011a",
    ("08_heat_sdrf_untyped_events", "stage_4_derived_relations"): "338b837255b54882",
    ("08_heat_sdrf_untyped_events", "stage_5_relations"): "338b837255b54882",
    ("08_heat_sdrf_untyped_events", "stage_6_edge_filters"): "338b837255b54882",
    ("08_heat_sdrf_untyped_events", "stage_7_relation_merging"): "fa2d103b805371d2",
    ("08_heat_sdrf_untyped_events", "representation"): "c4acd8ab30aaf187",
    ("09_fosr_drop_cap_walks", "stage_5_relations"): "d16ae7e1c28d6021",
    ("09_fosr_drop_cap_walks", "stage_6_edge_filters"): "c66da097643429b8",
    ("09_fosr_drop_cap_walks", "stage_7_relation_merging"): "c66da097643429b8",
    ("09_fosr_drop_cap_walks", "representation"): "69b32f3a6c013b29",
    ("10_untyped_merge_all", "representation"): "3bc7be8c2d281609",
}
WINDOW = dict(scheme="window", unit="step", window=3)
PROBE = '''from graphspec import OPERATIONS, Operation, Phase


def probe(graph, registry, settings, seed, limits, carry, held):
    return graph


def stated(registry):
    return ()


OPERATIONS.register(
    "probe.edited",
    Operation(
        "probe", "edited", (Phase("probe", probe, None, None),), stated, (), (), (), ("time",), None, None, VERSION, 1
    ),
)
'''


def identical(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def listed(found):
    started = time.perf_counter()
    if "neighbor_ids" not in found:
        return found
    ids, offsets = np.asarray(found["neighbor_ids"]), np.asarray(found["neighbor_offsets"])
    owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    result = dict(found, neighbor_ids=ids[np.lexsort((ids, owner))])
    LOG.debug("listed exit ids=%d seconds=%.4f", len(ids), time.perf_counter() - started)
    return result


def test_a_pipeline_is_stated_as_its_operations_in_order_and_read_back_as_stated(bank, tmp_path, statement):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = opened.registry
    cfg = support.configuration(registry, dict(view="snapshot"), {"time/window": 2}, 5, SPLITS)
    assert [entry["operation"] for entry in cfg["pipeline"]] == list(support.SEVEN)
    assert [step.entry() for step in opened.stated(cfg)[1]] == cfg["pipeline"]
    assert [
        (step["operation"], step["label"], step["version"]) for step in opened.identity(cfg)["pipeline"]
    ] == [(name, None, OPERATIONS.named(name).version) for name in support.SEVEN]
    lacking = {name: value for name, value in cfg.items() if name != "pipeline"}
    with pytest.raises(KeyError, match="the configuration states no pipeline"):
        opened.stated(lacking)
    with pytest.raises(KeyError, match="the configuration states no pipeline"):
        fit(bank, tmp_path, lacking, [], 8, time.perf_counter() + 30, state(statement))
    window = copy.deepcopy(cfg["pipeline"][1])
    wrong = [
        ("a list", dict(window), TypeError, "a pipeline is a list of its operations in the order they apply"),
        ("unknown", [dict(window, operation="graphspec.nothing")], KeyError, "no operation is registered"),
        ("twice", [window, window], ValueError, "the pipeline holds graphspec.window twice"),
        ("slash", [dict(window, label="a/b")], ValueError, "is a word without a slash or a dot, found 'a/b'"),
        ("same", [dict(window, label="x"), dict(window, label="x")], ValueError, "the pipeline holds x twice"),
        ("bare", [dict(operation="graphspec.window")], ValueError, "the entry 0 of the pipeline states no parameters"),
        (
            "lacking",
            [dict(window, parameters={"time/scheme": "window", "time/unit": "step"})],
            ValueError,
            "the entry graphspec.window of the pipeline states no time/window, which is active",
        ),
        (
            "inactive",
            [dict(window, parameters=dict(window["parameters"], **{"time/past_periods": 1}))],
            ValueError,
            "the entry graphspec.window of the pipeline states time/past_periods, which are inactive or unknown",
        ),
    ]
    for _, stated, error, message in wrong:
        with pytest.raises(error, match=message):
            pipeline.resolved(registry, stated)
    labelled = [window, dict(window, label="again")]
    assert [step.name for step in pipeline.resolved(registry, labelled)] == ["graphspec.window", "again"]
    LOG.info("stated pipeline exit seconds=%.3f", time.perf_counter() - started)


def test_a_search_draws_which_operations_are_present_and_their_parameters_never_their_order(bank):
    started = time.perf_counter()
    registry = Deriver(bank, "cpu", support.SEED, RESIDENT).registry
    template = support.template()
    names = [parameter.name for parameter in pipeline.space(registry, template)]
    assert "pipeline/graphspec.window" in names and "pipeline/graphspec.window/time/window" in names
    grown = [dict(operation="graphspec.relations", label="first")] + template
    grown.append(dict(operation="graphspec.window", label="late"))
    held = [parameter.name for parameter in pipeline.space(registry, grown)]
    assert set(names) <= set(held) and {"pipeline/first", "pipeline/late/time/window"} <= set(held)
    drawn = [random_suggest(registry, seed, seed, {}, 600, template)[0]["pipeline"] for seed in range(6)]
    orders = {tuple(entry["operation"] for entry in stated) for stated in drawn}
    assert len(orders) > 1
    assert all(list(order) == [name for name in support.SEVEN if name in order] for order in orders)
    assert all(pipeline.entries(template, pipeline.flattened(template, stated)) == stated for stated in drawn)
    bounds = {"pipeline/graphspec.derive": False, "pipeline/graphspec.window": True}
    bounds.update({"pipeline/graphspec.window/time/" + name: value for name, value in WINDOW.items()})
    fixed = random_suggest(registry, 3, 3, bounds, 600, template)[0]["pipeline"]
    assert "graphspec.derive" not in [entry["operation"] for entry in fixed]
    (window,) = [entry for entry in fixed if entry["operation"] == "graphspec.window"]
    assert window["parameters"] == {"time/" + name: value for name, value in WINDOW.items()}
    reordered = [entry for entry in reversed(drawn[0])]
    if len(reordered) > 1:
        with pytest.raises(ValueError, match="the pipeline orders its entries"):
            pipeline.flattened(template, reordered)
    LOG.info("drawn pipelines exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture
def sighted():
    seen = {}

    def blind(graph, registry, settings, seed, limits, carry, held):
        settings.split("trained")
        return graph

    def looking(graph, registry, settings, seed, limits, carry, held):
        seen["trained"] = settings.split("trained")
        settings.split("scored")
        return graph

    found = [
        Operation(
            "test",
            "blind",
            (Phase("blind", blind, None, None),),
            lambda registry: (),
            (),
            (),
            (),
            ("time",),
            None,
            None,
            1,
            1,
        ),
        Operation(
            "test", "looking", (Phase("looking", looking, None, None),), lambda registry: (), (), ("trained",), (),
            ("time",), None, None, 1, 1,
        ),
    ]
    for operation in found:
        OPERATIONS.register(operation.name, operation)
    yield seen
    for operation in found:
        del OPERATIONS.held[operation.name]


def test_an_operation_receives_only_the_splits_it_states_it_may_read(bank, sighted):
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    registry, partitions = opened.fitted(cfg), opened.partitions(cfg)
    dense = support.shown(opened, 0).dense()
    blind = pipeline.resolved(registry, [dict(operation="test.blind", parameters={})])
    with pytest.raises(
        KeyError, match="the operation test.blind reads the split trained, which it does not state it may read; "
        "it states none"
    ):
        segments.whole(dense, {}, registry, blind, support.SEED, GraphLimits("cpu"), partitions)
    looking = pipeline.resolved(registry, [dict(operation="test.looking", parameters={})])
    with pytest.raises(KeyError, match="the operation test.looking reads the split scored, which it does not"):
        segments.whole(dense, {}, registry, looking, support.SEED, GraphLimits("cpu"), partitions)
    rows = np.asarray(dense.label_row, np.int64)
    wanted = np.unique(np.asarray(dense.label_task)[partitions[rows] == 0])
    assert len(wanted) and np.array_equal(sighted["trained"], wanted)


def test_the_key_of_a_variant_holds_versions_and_no_file_and_editing_a_file_leaves_it(bank, tmp_path, monkeypatch):
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    key = opened.identity(cfg)
    assert sorted(key) == [
        "added", "device", "entries", "groups", "pipeline", "schema", "seeds", "settings", "source", "splits",
        "versions",
    ]
    assert key["schema"] == opened.manifest["spec_sha256"] and key["seeds"] == dict(derivation=support.SEED)
    assert key["added"] == []
    before = opened.variant(cfg)[0]
    seen = dict(cfg, splits=dict(SPLITS, sees=dict(train=["without_scored"], validation=[])))
    assert opened.variant(seen)[0] == Deriver(bank, "cpu", support.SEED, RESIDENT).variant(seen)[0] != before
    assert opened.holds(seen) - opened.holds(cfg) == {(SEES.label, "without_scored")}
    rows = [opened.arrays(0, held).read("row_index") for held in (cfg, seen)]
    assert not identical(*rows) and opened.variant(cfg)[0] == before
    other = Deriver(bank, "cpu", support.SEED + 1, RESIDENT)
    assert other.identity(cfg)["seeds"] == dict(derivation=support.SEED + 1) and other.variant(cfg)[0] != before
    for wrong in (-1, 2**64, 1.5, None):
        with pytest.raises(ValueError, match="the seed of derivation is a whole number from 0 to %d" % (2**64 - 1)):
            Deriver(bank, "cpu", wrong, RESIDENT)
    edges = OPERATIONS.named("graphspec.edges")
    monkeypatch.setattr(edges, "version", edges.version + 1)
    assert Deriver(bank, "cpu", support.SEED, RESIDENT).variant(cfg)[0] != before
    monkeypatch.undo()
    assert Deriver(bank, "cpu", support.SEED, RESIDENT).variant(cfg)[0] == before
    module = tmp_path / "edited_probe.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    stated = dict(cfg, pipeline=cfg["pipeline"] + [dict(operation="probe.edited", parameters={})])
    roots = []
    try:
        for version, body in ((1, "graph"), (1, "graph.replace()"), (2, "graph.replace()")):
            OPERATIONS.held.pop("probe.edited", None)
            module.write_text(PROBE.replace("VERSION", str(version)).replace("return graph", "return " + body))
            if "edited_probe" in sys.modules:
                importlib.reload(sys.modules["edited_probe"])
            else:
                importlib.import_module("edited_probe")
            roots.append(Deriver(bank, "cpu", support.SEED, RESIDENT).variant(stated)[0])
    finally:
        OPERATIONS.held.pop("probe.edited", None)
        sys.modules.pop("edited_probe", None)
    assert roots[0] == roots[1] != roots[2]


@timed("pipelines.unseeing")
def unseeing(facts, registry, held):
    return everything(facts, registry, held)


unseeing.version = 2


@timed("pipelines.ruled")
def ruled(registry):
    return (specs.choice("ruled/rule", NEGATIVES.names(), "degree"),)


@timed("pipelines.ruling")
def ruling(registry):
    return (specs.registered("ruled/rule", NEGATIVES, "degree"),)


@timed("pipelines.unruled")
def unruled(registry):
    return ()


@timed("pipelines.chosen")
def chosen(graph, registry, settings, seed, limits, carry, held):
    NEGATIVES.named(settings["ruled/rule"])
    return graph


@timed("pipelines.fixed")
def fixed(graph, registry, settings, seed, limits, carry, held):
    NEGATIVES.named("hops")
    return graph


READERS = dict(
    chosen=(chosen, ruled, "degree"),
    fixed=(fixed, unruled, "hops"),
    stated=(chosen, ruling, None),
)


def test_every_entry_a_variant_reads_states_a_version_and_the_keys_of_the_variant_and_its_views_hold_it(bank):
    started = time.perf_counter()
    unversioned = Representation("test.probe", "encode", uncounted, unlooking, running, None, None, unchecked, None)
    with pytest.raises(ValueError, match="the representation test.probe states no version"):
        REPRESENTATIONS.register("test.probe", unversioned)
    with pytest.raises(ValueError, match="the choice among several facts test.middle states no version"):
        SEVERAL.register("test.middle", lambda starts, stops: starts)
    assert "test.probe" not in REPRESENTATIONS.names() and "test.middle" not in SEVERAL.names()
    found = {}
    for version, seeing in ((1, everything), (2, unseeing)):
        REPRESENTATIONS.register(
            "test.probe",
            Representation("test.probe", "encode", uncounted, unlooking, running, None, None, unchecked, version),
        )
        NEGATIVES.register("test.parity", Rule(parities, parity, version))
        SEES.register("test.sees", seeing)
        try:
            opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
            splits = dict(SPLITS, sees=dict(train=["test.sees"], validation=[]))
            values = {"negatives/of": "link:visits", "negatives/rule": "test.parity", "negatives/count": 2}
            model = {"view": "snapshot", "representations/test.probe": True}
            cfg = support.configuration(
                opened.registry, model, values, 5, splits, support.SEVEN + ("graphspec.negatives",)
            )
            found[version] = opened.identity(cfg)["entries"], opened.variant(cfg)[0], opened.viewed(cfg).folder
        finally:
            del REPRESENTATIONS.held["test.probe"], NEGATIVES.held["test.parity"], SEES.held["test.sees"]
    for version in (1, 2):
        entries = {tuple(entry) for entry in found[version][0]}
        assert {
            ("representation", "test.probe", version),
            ("rule of negative sampling", "test.parity", version),
            ("rule of what a split may see", "test.sees", version),
            ("choice of steps", "tasks", 1),
            ("column type", "number", 1),
            ("kind of target", "number", 1),
        } <= entries
    assert found[1][1] != found[2][1] and found[1][2] != found[2][2]
    for name, (function, parameters, read) in READERS.items():
        operation = Operation("test", name, (Phase(name, function, None, None),), parameters, (), (), (), (), None, None, 1, 1)
        OPERATIONS.register(operation.name, operation)
        try:
            opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
            cfg = support.configuration(
                opened.registry, dict(view="snapshot"), support.several(opened.registry), 5, SPLITS, (operation.name,)
            )
            if read is None:
                assert ["rule of negative sampling", "degree", 1] in opened.identity(cfg)["entries"]
                assert len(opened.arrays(opened.graphs(cfg)[0], cfg))
                continue
            with pytest.raises(
                ValueError,
                match="the operation %s reads the rule of negative sampling %s, which the key of the variant it "
                "derives does not hold" % (operation.name, read),
            ):
                opened.arrays(opened.graphs(cfg)[0], cfg)
        finally:
            del OPERATIONS.held[operation.name]
    LOG.info("versioned entries exit seconds=%.3f", time.perf_counter() - started)


def test_the_states_of_added_columns_are_fitted_again_when_a_version_the_key_of_a_variant_holds_is_raised(
    bank, monkeypatch
):
    started = time.perf_counter()
    values = {"measures/degree": True, "measures/encoder": "standard"}
    names = support.SEVEN + ("graphspec.measures",)
    fits, trained = [], Deriver.trained

    def counted(*arguments):
        fits.append(arguments[0])
        return trained(*arguments)

    monkeypatch.setattr(Deriver, "trained", counted)
    found, counts = [], []
    for raised in (None, (transform, "VERSION"), (type(ENCODERS.named("standard")), "version")):
        if raised is not None:
            monkeypatch.setattr(*raised, getattr(*raised) + 1)
        opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
        cfg = support.configuration(opened.registry, dict(view="snapshot"), values, 5, SPLITS, names)
        opened.fitted(cfg)
        Deriver(bank, "cpu", support.SEED, RESIDENT).fitted(cfg)
        found.append(sorted(path.name for path in (opened.viewed(cfg).folder / "added").glob("*.json")))
        counts.append(len(fits))
    assert "standard" not in {column["encoder"] for column in opened.registry.schema.columns.values()}
    assert counts[1:] == [counts[0] + 1, counts[0] + 2] and len(found[0]) >= 1
    assert set(found[0]) < set(found[1]) < set(found[2])
    steps, calls, added = opened.stated(cfg)[1], [], Operation.added

    @timed("pipelines.walked")
    def walked(self, *arguments):
        calls.append(self.name)
        return added(self, *arguments)

    monkeypatch.setattr(Operation, "added", walked)
    key = opened.identity(cfg)
    assert sorted(calls) == sorted(step.operation.name for step in steps) and key["added"]
    LOG.info("states of added columns exit seconds=%.3f", time.perf_counter() - started)


def test_the_seven_in_todays_order_give_the_arrays_of_the_fixture_after_each_stage_and_the_representation(
    ground, grounded
):
    started = time.perf_counter()
    held, _ = fixture.configurations(ground)
    labels, pins = fixture.labels(ground), {}
    for entry in held:
        opened = Deriver(grounded[entry["source"]["name"]] / "bank", "cpu", support.SEED, RESIDENT)
        registry = support.fitted(opened)
        stated, rest = fixture.pipeline(opened.registry, entry)
        recorded = fixture.spec(entry["spec"])
        own = {name: value for step in stated for name, value in step["parameters"].items()}
        assert {name: value for name, value in own.items() if name in recorded} == {
            name: recorded[name] for name in own if name in recorded
        }
        steps = pipeline.resolved(registry, stated)
        chosen = tuple(entry["groups"])
        spec = specs.configured(specs.settings(opened.registry, chosen), rest)
        level, folder = entry["source"]["level"], ground / entry["folder"]
        stored = [fixture.arrays(folder / name, level) for name in [entry["files"]["input"]] + entry["files"]["stages"]]
        represented = fixture.arrays(folder / entry["files"]["representation"], level)
        names = [step.name for step in steps]
        assert len(names) == len(labels) and fixture.holds(names, len(names))
        found = {}
        for name in entry["graphs"]:
            graph = Dense(name, [], len(stored[0][name]["period"]), **stored[0][name])
            for index, step in enumerate(steps):
                graph = segments.whole(graph, {}, registry, (step,), support.SEED, GraphLimits("cpu"), None)
                if fixture.holds(names, index + 1):
                    stage = Path(entry["files"]["stages"][index]).stem
                    replaced = fixture.drawn(entry, stage)
                    assert set(fixture.differing(graph.arrays, stored[index + 1][name])) <= set(replaced), (
                        entry["name"],
                        step.name,
                    )
                    if replaced:
                        found.setdefault(stage, []).append(keyed.pinned(graph, registry, replaced))
            derivation = Derivation.begun(
                [graph], {}, registry, (), spec, chosen, support.SEED, GraphLimits("cpu"), None, 0
            )
            arrays = derivation.continued(graph.steps, None).arrays()
            now, then = listed(fixture.content(arrays)), listed(fixture.content(represented[name]))
            assert sorted(set(then) - set(now)) == sorted(GONE) and sorted(set(now) - set(then)) == sorted(ADDED)
            replaced = fixture.drawn(entry, "representation")
            differing = {key for key in set(now) & set(then) if not identical(now[key], then[key])}
            assert differing <= set(CHANGED) | set(replaced), (entry["name"], name, sorted(differing))
            assert level == "node" or not {"key", "task"} & differing
            if replaced:
                own = sorted(set(replaced) & set(arrays))
                found.setdefault("representation", []).append(keyed.pinned(graph, registry, own, arrays))
            if "events" not in replaced:
                events, held = np.asarray(now["events"]), np.asarray(then["events"])
                assert identical(np.delete(events, 7, 1), np.delete(held, 7, 1))
                assert identical(events[events[:, 3] != 2], held[events[:, 3] != 2])
        pins.update({(entry["name"], stage): keyed.combined(held) for stage, held in found.items()})
    assert pins == PIPELINED
    LOG.info("fixture pipeline exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture
def packed(monkeypatch):
    import torch

    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)


@pytest.mark.parametrize(
    "names,labels",
    [
        (tuple(reversed(support.SEVEN)), None),
        (support.SEVEN + ("graphspec.edges",), (None,) * 7 + ("again",)),
    ],
    ids=["reordered", "twice"],
)
def test_a_pipeline_in_another_order_or_with_an_operation_twice_completes_one_fit(
    bank, packed, tmp_path, statement, names, labels
):
    from tuning.families.model_spec import configuration, configured, groups

    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = opened.registry
    values = {"time/window": 2, "derive/hops": 2, "edges/visits/persistence": 2, "time/history": 4}
    stated, rest = support.pipelined(registry, values, names, labels)
    model = configured(registry, dict(view="snapshot"))
    graph = specs.configured(specs.settings(registry, groups(model)), rest)
    cfg = dict(configuration(registry, model, stated, graph, 5, SPLITS), epochs=1)
    blocks = opened.blocks(0, 4, cfg)
    fit(bank, tmp_path, cfg, blocks, 16, time.perf_counter() + statement["seconds"], state(statement))
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["config"]["pipeline"] == stated
    assert [step["operation"] for step in opened.identity(cfg)["pipeline"]] == list(names)
    assert saved["train_unique_rows"] == len(opened.members(cfg, 0))
    LOG.info("fit of a pipeline exit names=%d seconds=%.3f", len(names), time.perf_counter() - started)
