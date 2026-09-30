import copy
import importlib
import logging
import sys
import time

import fixture
import numpy as np
import pytest
import support

from conftest import SPLITS, Inline
from graphspec import OPERATIONS, Operation
from graphspec.budgets import GraphLimits
from graphspec.engine import transform
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.representations import groups as representations
from graphspec.views.graph import Dense
from tuning.bank.bank import Deriver
from tuning.fits.training import fit
from tuning.searches.selector import random_suggest
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("pipelines")
READS_EVERY_RELATION = dict(at=("09_fosr_drop_cap_walks", "derived relations"), commit="1bf1d1a")
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
)
CHANGED = dict(events="9249781", key="9249781", task="9249781", walks="c390954", period_walks="c390954")
WINDOW = dict(scheme="window", unit="step", window=3)
PROBE = '''from graphspec import OPERATIONS, Operation


def probe(graph, registry, settings, seed, limits, carry):
    return graph


def stated(registry):
    return ()


OPERATIONS.register(
    "probe.edited", Operation("probe", "edited", probe, stated, (), (), (), ("time",), VERSION)
)
'''


def identical(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def test_a_pipeline_is_stated_as_its_operations_in_order_and_read_back_as_stated(bank, tmp_path, statement):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
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
    registry = Deriver(bank, "cpu", RESIDENT).registry
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

    def blind(graph, registry, settings, seed, limits, carry):
        settings.split("trained")
        return graph

    def looking(graph, registry, settings, seed, limits, carry):
        seen["trained"] = settings.split("trained")
        settings.split("scored")
        return graph

    found = [
        Operation("test", "blind", blind, lambda registry: (), (), (), (), ("time",), 1),
        Operation("test", "looking", looking, lambda registry: (), (), ("trained",), (), ("time",), 1),
    ]
    for operation in found:
        OPERATIONS.register(operation.name, operation)
    yield seen
    for operation in found:
        del OPERATIONS.held[operation.name]


def test_an_operation_receives_only_the_splits_it_states_it_may_read(bank, sighted):
    opened = Deriver(bank, "cpu", RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    registry, partitions = opened.fitted(cfg), opened.partitions(cfg)
    dense = support.shown(opened, 0).dense()
    blind = pipeline.resolved(registry, [dict(operation="test.blind", parameters={})])
    with pytest.raises(
        KeyError, match="the operation test.blind reads the split trained, which it does not state it may read; "
        "it states none"
    ):
        transform.derive(dense, registry, blind, GraphLimits("cpu"), partitions)
    looking = pipeline.resolved(registry, [dict(operation="test.looking", parameters={})])
    with pytest.raises(KeyError, match="the operation test.looking reads the split scored, which it does not"):
        transform.derive(dense, registry, looking, GraphLimits("cpu"), partitions)
    rows = np.asarray(dense.label_row, np.int64)
    wanted = np.unique(np.asarray(dense.label_task)[partitions[rows] == 0])
    assert len(wanted) and np.array_equal(sighted["trained"], wanted)


def test_the_key_of_a_variant_holds_versions_and_no_file_and_editing_a_file_leaves_it(bank, tmp_path, monkeypatch):
    opened = Deriver(bank, "cpu", RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    key = opened.identity(cfg)
    assert sorted(key) == [
        "device", "groups", "pipeline", "schema", "seeds", "settings", "source", "splits", "versions"
    ]
    assert key["schema"] == opened.manifest["spec_sha256"] and key["seeds"] == dict(derivation=transform.SEED)
    before = opened.variant(cfg)[0]
    edges = OPERATIONS.named("graphspec.edges")
    monkeypatch.setattr(edges, "version", edges.version + 1)
    assert Deriver(bank, "cpu", RESIDENT).variant(cfg)[0] != before
    monkeypatch.undo()
    assert Deriver(bank, "cpu", RESIDENT).variant(cfg)[0] == before
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
            roots.append(Deriver(bank, "cpu", RESIDENT).variant(stated)[0])
    finally:
        OPERATIONS.held.pop("probe.edited", None)
        sys.modules.pop("edited_probe", None)
    assert roots[0] == roots[1] != roots[2]


def test_the_seven_in_todays_order_give_the_arrays_of_the_fixture_after_each_stage_and_the_representation(
    ground, grounded
):
    started = time.perf_counter()
    held, _ = fixture.configurations(ground)
    labels = fixture.labels(ground)
    for entry in held:
        opened = Deriver(grounded[entry["source"]["name"]] / "bank", "cpu", RESIDENT)
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
        drawn = fixture.arrays(folder / entry["files"]["representation"], level)
        for name in entry["graphs"]:
            graph, reached = Dense(name, [], len(stored[0][name]["period"]), **stored[0][name]), False
            for index, (label, step) in enumerate(zip(labels, steps, strict=True)):
                graph = transform.derive(graph, registry, (step,), GraphLimits("cpu"), None)
                reached = reached or (entry["name"], label) == READS_EVERY_RELATION["at"]
                if not reached:
                    assert fixture.differing(graph.arrays, stored[index + 1][name]) == [], (entry["name"], label)
            arrays = representations.build(graph, registry, spec, chosen, transform.SEED, GraphLimits("cpu"))
            now, then = fixture.content(arrays), fixture.content(drawn[name])
            assert sorted(set(then) - set(now)) == sorted(GONE) and sorted(set(now) - set(then)) == sorted(ADDED)
            if reached:
                continue
            differing = {key for key in set(now) & set(then) if not identical(now[key], then[key])}
            assert differing <= set(CHANGED), (entry["name"], name, sorted(differing))
            assert level == "node" or not {"key", "task"} & differing
            events, held = np.asarray(now["events"]), np.asarray(then["events"])
            assert identical(np.delete(events, 7, 1), np.delete(held, 7, 1))
            assert identical(events[events[:, 3] != 2], held[events[:, 3] != 2])
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
    opened = Deriver(bank, "cpu", RESIDENT)
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
