import logging
import re
import time

import numpy as np
import pytest
import support
from registered import HUB, KEPT, LINKED, RUNNING

from conftest import SPLITS
from graphspec import OPERATIONS, REPRESENTATIONS, Representation
from graphspec.budgets import GraphLimits
from graphspec.engine import segments
from graphspec.engine.edits import Edit, edited, sliced
from graphspec.engine.transform import SEED
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.operations import contract, operation, pipeline
from graphspec.operations.consecutive import CONSECUTIVE
from graphspec.operations.nearby import NEARBY, measured, placed
from graphspec.operations.pipeline import resolved
from graphspec.operations.rows import keyed_relations
from graphspec.representations.registry import FAMILIES, families
from graphspec.schema.added import Added
from graphspec.views.graph import Graph
from tuning.bank.bank import Deriver
from tuning.families.model_spec import DERIVES, derived
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("segments")
SHIPPED = dict(structure=("coupling", "events", "mixed"), sample=("neighbors", "walks"), encode=())
ADDED = (LINKED, KEPT, RUNNING)
LIMITS = GraphLimits("cpu")
REFUSED = (
    ("graphspec.nodes", {"nodes/item/keep": False}, "item"),
    ("graphspec.nodes", {"nodes/item/filter": "degree"}, "item"),
    ("graphspec.derive", {"derive/project:owns:targets:item:item": True}, "item"),
    ("graphspec.derive", {"derive/complete/item": True}, "item"),
    ("graphspec.derive", {"derive/complete/person": True, "derive/complete_within/item": True}, "item"),
    ("graphspec.nearby", {"nearby/kinds": "person:item", "nearby/encoder": "standard"}, "item"),
    ("graphspec.consecutive", {"consecutive/pairing": "near:visits:targets"}, "place"),
    ("graphspec.neighbourhood", {"neighbourhood/around": "item"}, "item"),
    ("graphspec.contract", {"contract/grouping": "item:v1"}, "item"),
)
REMOVERS = (("graphspec.contract", None), ("graphspec.nodes", "gone"))


@pytest.fixture(scope="module")
def opened(bank):
    held = Deriver(bank, "cpu", RESIDENT)
    names = [graph["name"] for graph in held.manifest["graphs"]]
    return held, support.fitted(held), support.shown(held, 0).dense(), names


@timed("added.observed")
def observed(graph):
    tokens, table = np.asarray(graph.entity_token), np.asarray(graph.x_table)
    return {
        (int(graph.step_time[step]), int(tokens[entity])): table[reference].tobytes()
        for step, entity, reference in zip(graph.obs_step.tolist(), graph.obs_entity.tolist(), graph.obs_ref.tolist())
    }


@timed("added.stated")
def stated(graph, registry):
    tokens, both = np.asarray(graph.entity_token), keyed_relations(registry)[1]
    found = {}
    for row in range(len(graph.fact_step)):
        code = int(graph.fact_code[row])
        ends = int(tokens[graph.fact_source[row]]), int(tokens[graph.fact_target[row]])
        ends = tuple(sorted(ends)) if both[code] else ends
        reference = int(graph.fact_ref[row])
        values = b"" if reference < 0 else np.asarray(graph.value_table)[reference].tobytes()
        key = (int(graph.step_time[graph.fact_step[row]]), code) + ends
        found.setdefault(key, []).append((bool(graph.fact_present[row]), values))
    return found


def listed(found):
    return sorted((key, row) for key, rows in found.items() for row in rows)


@pytest.fixture
def represented():
    for entry in ADDED:
        REPRESENTATIONS.register(entry.name, entry)
    yield
    for entry in ADDED:
        del REPRESENTATIONS.held[entry.name]


@pytest.fixture
def hubbed():
    OPERATIONS.register(HUB.name, HUB)
    yield
    del OPERATIONS.held[HUB.name]


@timed("added.counted")
def counted(view, place, bound):
    structural = ~np.asarray(view.obs_anchor, bool)
    kinds = np.asarray(view.kinds)[np.asarray(view.obs_entity)]
    counts = np.bincount(np.asarray(view.obs_step)[structural & (kinds == place)], minlength=view.steps)[:bound]
    return counts[counts > 0]


def test_an_operation_adds_a_kind_a_relation_and_a_column_encoded_from_the_trained_views_alone(bank, hubbed):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS, (HUB.name,))
    registry, count, inward = opened.fitted(cfg), "hub.count", "hub.in"
    assert {entry["name"]: entry["adds"] for entry in operation.listed(opened.registry)}[HUB.name] is True
    assert registry.nodes == opened.registry.nodes + ("hub",) and registry.declared == opened.registry.nodes
    edge = registry.by_name[inward]
    assert (edge.pairs, edge.attributes, edge.origin, edge.code) == ((("place", "hub"),), (count,), ("added", "hub"), len(registry.edges) - 1)
    assert registry.valued[-1] == inward and registry.schema.columns[count] == dict(type="number", encoder="standard")
    assert "hub adds the column hub.count, number by standard; the kind of node hub" in registry.schema.text(registry.encoding)
    declared = dict(
        entry="hub",
        columns={count: dict(type="number", encoder="standard")},
        kinds=dict(hub=dict(columns=[count])),
        placed={},
        relations={inward: dict(pairs=[["place", "hub"]], directed=True, columns=[count])},
        joins={},
    )
    assert opened.identity(cfg)["added"] == [declared] and registry.declaration()["added"] == [declared]
    viewed, partitions, place = opened.viewed(cfg), opened.partitions(cfg), registry.kind("place")
    found = {True: [], False: []}
    for graph in viewed.graphs:
        view = Graph.load(viewed.path(graph["index"])).dense()
        rows, places = viewed.labels[graph["index"]]
        moments = viewed.row_steps[graph["index"]][places[partitions[rows] == 0]]
        for key, bound in ((True, int(moments.max(initial=-1)) + 1), (False, view.steps)):
            counts = counted(view, place, bound)
            found[key].append(np.repeat(counts, counts + 1))
    trained, every = (np.concatenate(found[key]).astype(np.float64) for key in (True, False))
    state = registry.encoding.states[count]
    assert state == pytest.approx(dict(mean=trained.mean(), sd=trained.std(ddof=1)), rel=1e-6)
    assert abs(state["mean"] - every.mean()) > 0.01
    arrays = opened.arrays(0, cfg)
    x, kinds = np.asarray(arrays["x_table"]), np.asarray(arrays["entity_kind"])
    hubs = kinds[np.asarray(arrays["entity"])] == registry.kind("hub")
    values, flags = registry.encoding.slots(count)
    shown = x[np.asarray(arrays["x_ref"])[hubs]].astype(np.float64)
    view = Graph.load(viewed.path(0)).dense()
    counts = counted(view, place, view.steps)
    assert x.shape[1] == registry.features and (shown[:, flags] == 1).all()
    assert np.allclose(shown[:, values[0]] * state["sd"] + state["mean"], counts, atol=0.05)
    attributes = np.asarray(arrays["attributes_" + inward], np.float64)
    assert sorted(attributes[:, 0].tolist()) == sorted(np.repeat(counts, counts).astype(np.float64).tolist())
    with pytest.raises(ValueError, match="adds the columns lost and places them nowhere"):
        opened.registry.grown("bad", Added({"lost": dict(type="number", encoder="value")}, {}, {}, {}, {}), None)
    with pytest.raises(ValueError, match="adds the kind of node place, a name the registry holds"):
        opened.registry.grown("bad", Added({}, {"place": dict(columns=(), tokens=len)}, {}, {}, {}), None)
    LOG.info("added kind relation column exit seconds=%.3f", time.perf_counter() - started)


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


def test_a_merge_takes_the_graph_it_names_on_keys_at_the_steps_of_the_graph_it_changes(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    second = support.beside(held, 0, names[1])
    assert np.array_equal(second.step_time, graph.step_time) and second.name.startswith(names[1])
    for prefer in ("first", "second"):
        entry = dict(operation="graphspec.merge", graph=names[1], parameters={"merge/prefer": prefer})
        merged = segments.whole(graph, {names[1]: second}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
        ordered = (second, graph) if prefer == "first" else (graph, second)
        assert observed(merged) == {**observed(ordered[0]), **observed(ordered[1])}
        mine, theirs = stated(ordered[1], registry), stated(ordered[0], registry)
        assert listed(stated(merged, registry)) == listed({**theirs, **mine})
        assert set(np.asarray(merged.entity_token).tolist()) == set(np.r_[graph.entity_token, second.entity_token].tolist())
    with pytest.raises(ValueError, match="of graphspec.merge, which takes two graphs, names its second graph"):
        resolved(registry, [dict(operation="graphspec.merge", parameters={"merge/prefer": "first"})])
    with pytest.raises(ValueError, match="of graphspec.relations, which takes one graph, names a second graph"):
        resolved(registry, [dict(operation="graphspec.relations", graph=names[1], parameters={})])
    with pytest.raises(ValueError, match="take %s as second graphs, and it was given none" % names[1]):
        segments.whole(graph, {}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
    with pytest.raises(ValueError, match="shows other times than the graph"):
        segments.whole(graph, {names[1]: sliced(second, 0, 3)}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
    with pytest.raises(KeyError, match="the bank holds no graph named absent to show beside graph"):
        support.beside(held, 0, "absent")
    LOG.info("merge exit seconds=%.3f", time.perf_counter() - started)


def test_a_difference_removes_the_edges_the_graph_it_names_holds_at_the_same_steps(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    mine, removed = stated(graph, registry), {}
    for name in names[:2]:
        second = support.beside(held, 0, name)
        theirs = stated(second, registry)
        for match in ("keys", "values"):
            entry = dict(operation="graphspec.difference", graph=name, parameters={"difference/match": match})
            left = segments.whole(graph, {name: second}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
            kept = {
                key: [
                    row
                    for row in rows
                    if not row[0]
                    or not any(other[0] and (match == "keys" or other[1] == row[1]) for other in theirs.get(key, []))
                ]
                for key, rows in mine.items()
            }
            found = stated(left, registry)
            assert [row for row in listed(found) if row[1][0]] == [row for row in listed(kept) if row[1][0]]
            assert {row for row in listed(found) if not row[1][0]} <= {row for row in listed(mine) if not row[1][0]}
            assert observed(left) == observed(graph)
            removed[name, match] = len(listed(mine)) - len(listed(found))
    assert 0 < removed[names[1], "values"] <= removed[names[1], "keys"] < removed[names[0], "keys"]
    LOG.info("difference exit removed=%s seconds=%.3f", removed, time.perf_counter() - started)


@timed("added.added")
def added(found, graph, code):
    tokens = np.asarray(graph.entity_token)
    rows = [
        (int(found.step_time[step]), int(tokens[source]), int(tokens[target]))
        for step, source, target, kind in zip(
            found.fact_step.tolist(), found.fact_source.tolist(), found.fact_target.tolist(), found.fact_code.tolist()
        )
        if kind == code
    ]
    before = {
        (int(graph.step_time[step]), int(tokens[source]), int(tokens[target]))
        for step, source, target, kind in zip(
            graph.fact_step.tolist(), graph.fact_source.tolist(), graph.fact_target.tolist(), graph.fact_code.tolist()
        )
        if kind == code
    }
    return sorted(row for row in rows if row not in before), before


@timed("added.near")
def near(graph, rows, values, flags):
    table, tokens = np.asarray(graph.x_table), np.asarray(graph.entity_token)
    rows = rows[np.all(table[graph.obs_ref[rows]][:, flags] > 0, axis=1)]
    return table[graph.obs_ref[rows]][:, values].astype(np.float64), tokens[graph.obs_entity[rows]].astype(np.uint64)


def test_nearby_adds_a_relation_whose_edges_hold_the_distance_within_which_rows_lie_or_are_nearest(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    place, person = registry.kind("place"), registry.kind("person")
    assert measured(registry, "place", "place") == ("v0", "v1", "v2", "v3")
    assert {"place:place", "person:place", "place:person"} <= set(NEARBY.stated(registry)[0].options)
    values, flags = placed(registry, measured(registry, "place", "place"))
    for kinds, rule, stated_rule, count in (
        ("place:place", "threshold", {"nearby/threshold": 1.5}, None),
        ("place:place", "nearest", {"nearby/k": 3}, 3),
        ("person:place", "nearest", {"nearby/k": 2}, 2),
    ):
        parameters = {"nearby/kinds": kinds, "nearby/rule": rule, "nearby/encoder": "standard", **stated_rule}
        steps = resolved(registry, [dict(operation="graphspec.nearby", parameters=parameters)])
        grown = pipeline.grown(registry, steps, None)
        edge = grown.by_name["nearby"]
        assert (edge.symmetric, edge.attributes, edge.origin) == (rule == "threshold", ("nearby.distance",), ("added", "nearby"))
        found = segments.whole(graph, {}, grown, steps, SEED, LIMITS, None)
        expected = set()
        for step in range(graph.steps):
            at = (graph.obs_step == step) & ~graph.obs_anchor
            ends = [
                near(graph, np.flatnonzero(at & (graph.kinds[graph.obs_entity] == kind)), values, flags)
                for kind in (person if kinds.startswith("person") else place, place)
            ]
            for first, origin in zip(*ends[0]):
                distance = np.sqrt(((ends[1][0] - first) ** 2).sum(1))
                others = sorted((distance[second], int(token)) for second, token in enumerate(ends[1][1]) if token != origin)
                chosen = others[:count] if rule == "nearest" else [pair for pair in others if pair[0] <= 1.5]
                expected |= {
                    (int(graph.step_time[step]),) + ((int(origin), token) if rule == "nearest" else tuple(sorted((int(origin), token))))
                    + (np.float32(length),)
                    for length, token in chosen
                }
        tokens, rows = np.asarray(found.entity_token).astype(np.uint64), np.flatnonzero(found.fact_code == edge.code)
        assert found.fact_present[rows].all() and (found.fact_ref[rows] >= 0).all()
        new = {
            (int(found.step_time[found.fact_step[row]]),)
            + ((int(tokens[found.fact_source[row]]), int(tokens[found.fact_target[row]])) if rule == "nearest" else tuple(sorted((int(tokens[found.fact_source[row]]), int(tokens[found.fact_target[row]])))))
            + (np.float32(found.value_table[found.fact_ref[row], 0]),)
            for row in rows
        }
        assert new == expected and len(new) == len(rows) > 0
    flagged = support.columned("flagged", ["thing"], [], dict(open=dict(type="flag", encoder="value")))
    assert NEARBY.stated(flagged) == ()
    with pytest.raises(ValueError, match="by the distance between the values of the columns of numbers both kinds"):
        pipeline.grown(flagged, resolved(flagged, [dict(operation="graphspec.nearby", parameters={})]), None)
    LOG.info("nearby exit seconds=%.3f", time.perf_counter() - started)


@timed("added.held_facts")
def held_facts(graph, registry, rows, changed):
    tokens, both = np.asarray(graph.entity_token).astype(np.uint64), keyed_relations(registry)[1]
    found = []
    for row in np.asarray(rows).tolist():
        code, ends = int(graph.fact_code[row]), (int(tokens[graph.fact_source[row]]), int(tokens[graph.fact_target[row]]))
        reference = int(graph.fact_ref[row])
        values = () if reference < 0 else np.asarray(graph.value_table[reference], np.float32)
        values = changed[row] if row in changed else tuple("nan" if value != value else value for value in np.asarray(values).tolist())
        time_of = int(graph.step_time[graph.fact_step[row]])
        found.append((time_of, code) + (tuple(sorted(ends)) if both[code] else ends) + (bool(graph.fact_present[row]), values))
    return found


@timed("added.reduced")
def reduced(graph, aggregate):
    groups = {}
    for row in range(len(graph.fact_step)):
        key = (int(graph.fact_step[row]), bool(graph.fact_present[row]), int(graph.fact_code[row]))
        groups.setdefault(key + (int(graph.fact_source[row]), int(graph.fact_target[row])), []).append(row)
    kept, values = [], {}
    for key, rows in groups.items():
        if aggregate == "keep":
            kept += rows
            continue
        kept.append(rows[0] if aggregate == "first" else rows[-1])
        valued = [row for row in rows if key[1] and graph.fact_ref[row] >= 0]
        if aggregate in ("first", "last") or len(valued) < 2:
            continue
        held = np.asarray(graph.value_table, np.float64)[graph.fact_ref[valued]]
        found = dict(sum=np.sum, mean=np.mean, min=np.min, max=np.max)[aggregate](held, axis=0)
        values[rows[-1]] = tuple(found.astype(np.float32).tolist())
    return sorted(kept), values


def test_relation_forms_keep_or_reduce_parallel_edges_and_add_reverse_edges_and_self_loops(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    rows = np.flatnonzero(graph.fact_present & (graph.fact_code == registry.code("visits")))[:6]
    copies = np.r_[rows, rows[:3]]
    weights = np.asarray(graph.value_table)[graph.fact_ref[copies]] + np.r_[np.ones(6), np.full(3, 2.0)][:, None]
    nothing = dict(observations=dict(entity=[], step=[]), facts=dict(code=[], source=[], target=[], step=[]))
    stated_facts = {name: graph.arrays["fact_" + name][copies] for name in ("source", "target", "code", "step")}
    doubled = edited(
        graph,
        Edit(
            dict(kind=[], token=np.zeros(0, np.uint64)),
            nothing,
            dict(
                observations=dict(entity=[], step=[], x=np.zeros((0, graph.x_table.shape[1]))),
                facts=dict(stated_facts, present=np.ones(len(copies), bool), values=weights),
            ),
        ),
    )
    tokens = np.asarray(doubled.entity_token).astype(np.uint64)
    counts = {}
    for aggregate in ("keep", "first", "last", "sum", "mean", "min", "max"):
        stated = dict(types="typed", parallel=aggregate, reverse=True, loops=True)
        parameters = {"relation_forms/" + name: value for name, value in stated.items()}
        steps = resolved(registry, [dict(operation="graphspec.relation_forms", parameters=parameters)])
        grown = pipeline.grown(registry, steps, None)
        assert grown.by_name["relation_forms.visits"].pairs == (("place", "person"),)
        loops = grown.by_name["relation_forms.loops"]
        assert loops.symmetric and loops.pairs == tuple((kind, kind) for kind in registry.nodes)
        found = segments.whole(doubled, {}, grown, steps, SEED, LIMITS, None)
        kept, values = reduced(doubled, aggregate)
        expected = held_facts(doubled, grown, kept, values)
        expected += [
            (row[0], grown.code("relation_forms." + grown.by_code[row[1]].name), row[3], row[2]) + row[4:]
            for row in expected
            if grown.by_code[row[1]].name in ("owns", "member", "near", "visits", "located")
        ]
        expected += [
            (int(doubled.step_time[doubled.obs_step[row]]), loops.code)
            + (int(tokens[doubled.obs_entity[row]]),) * 2
            + (True, ())
            for row in np.flatnonzero(~doubled.obs_anchor)
        ]
        assert sorted(held_facts(found, grown, np.arange(len(found.fact_step)), {})) == sorted(expected)
        counts[aggregate] = len(found.fact_step)
    assert counts["keep"] == counts["first"] + 2 * len(copies)
    assert len({counts[name] for name in ("first", "last", "sum", "mean", "min", "max")}) == 1
    LOG.info("relation forms exit counts=%s seconds=%.3f", counts, time.perf_counter() - started)


@timed("added.cored")
def cored(nodes, pairs):
    found = {node: 0 for node in nodes}
    for least in range(1, len(pairs) + 2):
        alive = set(nodes)
        while True:
            degree = {node: 0 for node in alive}
            for a, b in pairs:
                if a in alive and b in alive:
                    degree[a] += 1
                    degree[b] += a != b
            gone = {node for node in alive if degree[node] < least}
            if not gone:
                break
            alive -= gone
        if not alive:
            return found
        found.update({node: least for node in alive})
    return found


@timed("added.structured")
def structured(graph):
    tokens, result = np.asarray(graph.entity_token).astype(np.uint64), {}
    for step in range(graph.steps):
        nodes = graph.obs_entity[graph.obs_step == step].tolist()
        chosen = (graph.fact_step == step) & graph.fact_present
        pairs = list(zip(graph.fact_source[chosen].tolist(), graph.fact_target[chosen].tolist()))
        degree = {node: sum(1 for a, b in pairs if node in (a, b)) for node in nodes}
        reached = {node: {node} for node in nodes}
        for a, b in pairs:
            joined = reached[a] | reached[b]
            for node in joined:
                reached[node] = joined
        cores = cored(nodes, pairs)
        result.update(
            {
                (int(graph.step_time[step]), int(tokens[node])): (degree[node], len(reached[node]), cores[node])
                for node in nodes
            }
        )
    return result


def test_measures_of_structure_are_columns_on_every_kind_of_node_at_every_step(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    parameters = {"measures/degree": True, "measures/components": True, "measures/cores": True, "measures/encoder": "logarithm"}
    steps = resolved(registry, [dict(operation="graphspec.measures", parameters=parameters)])
    grown = pipeline.grown(registry, steps, None)
    columns = ("measures.degree", "measures.components", "measures.cores")
    assert grown.schema.added_columns() == columns
    assert all(set(columns) <= set(held) for held in grown.schema.entities().values())
    declared = {entry["name"]: entry for entry in operation.listed(registry)}["graphspec.measures"]
    assert [(count["counts"], count["sets"], count["about"]) for count in declared["requires"]] == [
        ("degree", "stored", "stored"),
        ("components", "stored", "stored"),
        ("core numbers", "stored", "stored"),
    ]
    found = segments.whole(graph, {}, grown, steps, SEED, LIMITS, None)
    slots = [grown.encoding.slots(name) for name in columns]
    rows = np.asarray(found.x_table, np.float64)[found.obs_ref]
    tokens = np.asarray(found.entity_token).astype(np.uint64)
    assert all((rows[:, flags] == 1).all() for _, flags in slots)
    measured = {
        (int(found.step_time[step]), int(tokens[who])): tuple(int(row[values[0]]) for values, _ in slots)
        for step, who, row in zip(found.obs_step, found.obs_entity, rows)
    }
    expected = structured(graph)
    assert measured == expected and max(value[2] for value in expected.values()) > 1
    stated = np.asarray(found.x_table)[found.obs_ref][:, : registry.encoding.values]
    assert np.array_equal(stated, np.asarray(graph.x_table)[graph.obs_ref][:, : registry.encoding.values])
    LOG.info("measures exit observations=%d seconds=%.3f", len(measured), time.perf_counter() - started)


@timed("added.valued")
def valued(graph, row):
    reference = int(graph.fact_ref[row])
    found = () if reference < 0 else np.asarray(graph.value_table[reference], np.float32).tolist()
    return tuple("nan" if value != value else value for value in found)


@timed("added.rounded")
def rounded(graph, registry, column, missing):
    values, flags = registry.encoding.slots(column)
    table = np.array(graph.x_table)
    owners = np.full(len(table), -1, np.int64)
    owners[graph.obs_ref] = graph.obs_entity
    table[:, values] = np.round(table[:, values].astype(np.float64) / 2).astype(table.dtype)
    table[:, flags] = 1
    table[np.isin(owners, missing), flags[0]] = 0
    return graph.replace(x_table=table)


@timed("added.contracted")
def contracted(graph, grown, entry, aggregate):
    person, tokens = grown.kind("person"), np.asarray(graph.entity_token).astype(np.uint64)
    known, held = contract.blocks(grown, "v0", np.asarray(graph.x_table)[graph.obs_ref])
    members = (graph.kinds[graph.obs_entity] == person)
    grouping = np.zeros(len(members), np.uint64)
    grouping[members & known] = contract.group_tokens(entry, held[members & known])
    at = {(int(step), int(who)): int(grouping[row]) if members[row] and known[row] else (-1 if members[row] else int(tokens[who])) for row, (step, who) in enumerate(zip(graph.obs_step, graph.obs_entity))}
    observations, rows = set(), {}
    for row, (step, who) in enumerate(zip(graph.obs_step.tolist(), graph.obs_entity.tolist())):
        if not members[row]:
            observations.add((int(graph.step_time[step]), int(tokens[who]), graph.x_table[graph.obs_ref[row]].tobytes()))
        elif known[row]:
            rows.setdefault((step, int(grouping[row])), []).append(np.asarray(graph.x_table[graph.obs_ref[row]], np.float64))
    for (step, group), found in rows.items():
        found, made = np.stack(found), np.zeros(grown.features)
        for column in grown.schema.entities()["person"]:
            values, flags = grown.encoding.slots(column)
            have = found[np.all(found[:, flags] > 0, axis=1)][:, values]
            reduce = dict(mean=np.mean, sum=np.sum, min=np.min, max=np.max)[aggregate]
            made[values] = reduce(have, axis=0) if len(have) else 0.0
            made[flags] = len(have) > 0
        observations.add((int(graph.step_time[step]), group, made.astype(graph.x_table.dtype).tobytes()))
    facts = []
    for row in np.flatnonzero(graph.fact_present):
        step = int(graph.fact_step[row])
        ends = [at[(step, int(end))] for end in (graph.fact_source[row], graph.fact_target[row])]
        if -1 in ends:
            continue
        facts.append((int(graph.step_time[step]), int(graph.fact_code[row]), *ends, valued(graph, row)))
    return observations, sorted(facts), at


def test_contraction_makes_one_node_per_group_of_a_column_with_reduced_values_and_edges_that_follow(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    person = registry.kind("person")
    referred = set(graph.member_source.tolist()) | set(graph.member_target.tolist())
    quiet = [who for who in np.flatnonzero(graph.kinds == person).tolist() if who not in referred][:2]
    given = rounded(graph, registry, "v0", quiet)
    for aggregate in ("mean", "max"):
        parameters = {"contract/grouping": "person:v0", "contract/aggregate": aggregate, "contract/parallel": "keep"}
        steps = resolved(registry, [dict(operation="graphspec.contract", parameters=parameters)])
        grown = pipeline.grown(registry, steps, None)
        assert grown.nodes[-1] == "contract" and grown.schema.entities()["contract"] == registry.schema.entities()["person"]
        assert grown.by_name["visits"].pairs == (("person", "place"), ("contract", "place"))
        assert grown.by_name["member"].pairs == (("person", "team"), ("contract", "team"))
        found = segments.whole(given, {}, grown, steps, SEED, LIMITS, None)
        observations, facts, at = contracted(given, grown, "contract", aggregate)
        tokens = np.asarray(found.entity_token).astype(np.uint64)
        assert {
            (int(found.step_time[step]), int(tokens[who]), found.x_table[reference].tobytes())
            for step, who, reference in zip(found.obs_step, found.obs_entity, found.obs_ref)
        } == observations
        ends = (found.fact_source, found.fact_target)
        assert sorted(
            (int(found.step_time[found.fact_step[row]]), int(found.fact_code[row]))
            + tuple(int(tokens[end[row]]) for end in ends)
            + (valued(found, row),)
            for row in np.flatnonzero(found.fact_present)
        ) == facts
        moments = given.task_step[given.member_task]
        assert [int(tokens[who]) for who in found.member_source] == [
            at[(int(step), int(who))] for step, who in zip(moments, given.member_source)
        ]
        silent = np.asarray(given.entity_token)[quiet]
        assert not np.isin(tokens[found.obs_entity], silent).any() and (found.kinds[found.obs_entity] != person).all()
    summed = dict(parameters, **{"contract/parallel": "sum"})
    steps = resolved(registry, [dict(operation="graphspec.contract", parameters=summed)])
    fewer = segments.whole(given, {}, grown, steps, SEED, LIMITS, None)
    assert 0 < fewer.fact_present.sum() < found.fact_present.sum()
    task = int(np.flatnonzero(graph.kinds[graph.member_source] == person)[0])
    lacking = rounded(graph, registry, "v0", quiet + [int(graph.member_source[task])])
    with pytest.raises(ValueError, match="so graphspec.contract puts it in no group and the reference cannot follow it"):
        segments.whole(lacking, {}, grown, steps, SEED, LIMITS, None)
    bare = support.columned("bare", ["thing"], [], {})
    assert contract.CONTRACT.stated(bare) == ()
    with pytest.raises(ValueError, match="by one of the columns the bank states for it, and the bank bare states no"):
        pipeline.grown(bare, resolved(bare, [dict(operation="graphspec.contract", parameters={})]), None)
    LOG.info("contraction exit seconds=%.3f", time.perf_counter() - started)


@timed("added.sessions")
def sessions(graph, registry, pairing, gap):
    place, tokens = registry.kind("place"), np.asarray(graph.entity_token).astype(np.uint64)
    structural = ~np.asarray(graph.obs_anchor, bool)
    seen = {(int(step), int(who)) for step, who in zip(graph.obs_step[structural], graph.obs_entity[structural])}
    links = sorted((step, -1, who) for step, who in seen if graph.kinds[who] == place)
    if pairing != "near":
        chosen = graph.fact_present & (graph.fact_code == registry.code("visits"))
        links = sorted(
            {
                (int(step), int(owner), int(who))
                for step, owner, who in zip(graph.fact_step[chosen], graph.fact_source[chosen], graph.fact_target[chosen])
            }
        )
    first = {}
    for step, owner, who in links:
        first.setdefault((owner, who), step)
    groups = {}
    for (owner, who), step in first.items():
        groups.setdefault(owner, []).append((int(graph.step_time[step]), int(tokens[who]), who, step))
    expected = set()
    for rows in groups.values():
        rows.sort()
        for (moment, token, who, _), (later, other, _, step) in zip(rows, rows[1:]):
            if later - moment <= gap and (step, who) in seen:
                expected.add((int(graph.step_time[step]), token, other))
    return expected


def test_consecutive_rows_of_a_group_are_joined_unless_a_gap_in_time_cuts_them(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    options = CONSECUTIVE.stated(registry)[0].options
    assert {"near", "near:visits:targets", "near:near:sources", "near:located:targets"} <= set(options)
    counts = {}
    for pairing in ("near", "near:visits:targets"):
        for gap in (1.0, 1e9):
            entry = dict(
                operation="graphspec.consecutive",
                parameters={"consecutive/pairing": pairing, "consecutive/gap": gap},
            )
            found = segments.whole(graph, {}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
            new, before = added(found, graph, registry.code("near"))
            assert new == sorted(sessions(graph, registry, pairing, gap) - before)
            counts[pairing, gap] = len(new)
    assert 0 < counts["near:visits:targets", 1.0] < counts["near:visits:targets", 1e9]
    assert counts["near", 1.0] <= counts["near", 1e9]
    plain = support.registry("plain", ["thing"], [])
    carry, dense = segments.opening(graph, {}, registry)
    with pytest.raises(ValueError, match="fills a relation stated without columns from a kind of node to itself"):
        CONSECUTIVE(dense, plain, {}, SEED, LIMITS, carry, CONSECUTIVE.initial(plain, {}), None)
    LOG.info("consecutive exit counts=%s seconds=%.3f", counts, time.perf_counter() - started)


@timed("added.surroundings")
def surroundings(graph, registry, around, hops):
    tokens, kinds = np.asarray(graph.entity_token), np.asarray(graph.kinds)
    observations, facts = set(), set()
    for step in range(graph.steps):
        moment = int(graph.step_time[step])
        observed = set(graph.obs_entity[graph.obs_step == step].tolist())
        tasks = np.flatnonzero(graph.task_step == step)
        chosen = np.isin(graph.member_task, tasks)
        reached = {int(who) for who in np.r_[graph.member_source[chosen], graph.member_target[chosen]] if who >= 0}
        if around != "references":
            reached |= {who for who in observed if kinds[who] == registry.kind(around)}
        reached &= observed
        present = (graph.fact_step == step) & graph.fact_present
        pairs = list(zip(graph.fact_source[present].tolist(), graph.fact_target[present].tolist()))
        for _ in range(hops):
            reached |= {b for a, b in pairs if a in reached} | {a for a, b in pairs if b in reached}
        observations |= {(moment, int(tokens[who])) for who in reached}
        facts |= {
            (moment, int(code), int(tokens[a]), int(tokens[b]))
            for (a, b), code in zip(pairs, graph.fact_code[present].tolist())
            if a in reached and b in reached
        }
    return observations, facts


def test_a_neighbourhood_keeps_what_lies_within_a_number_of_hops_of_its_nodes_at_each_step(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    sizes = {}
    for around, hops in (("references", 0), ("references", 2), ("team", 1)):
        entry = dict(
            operation="graphspec.neighbourhood",
            parameters={"neighbourhood/hops": hops, "neighbourhood/around": around},
        )
        found = segments.whole(graph, {}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
        tokens = np.asarray(found.entity_token)
        observations = {
            (int(found.step_time[step]), int(tokens[who])) for step, who in zip(found.obs_step, found.obs_entity)
        }
        present = np.asarray(found.fact_present)
        facts = {
            (int(found.step_time[step]), int(code), int(tokens[a]), int(tokens[b]))
            for step, code, a, b in zip(
                found.fact_step[present], found.fact_code[present], found.fact_source[present], found.fact_target[present]
            )
        }
        assert (observations, facts) == surroundings(graph, registry, around, hops)
        sizes[around, hops] = len(observations), len(facts)
    assert sizes["references", 0][1] < sizes["references", 2][1] and sizes["references", 2][0] < len(graph.obs_step)
    LOG.info("neighbourhood exit sizes=%s seconds=%.3f", sizes, time.perf_counter() - started)


@timed("added.entry")
def entry(registry, name, values, label, graph):
    entries, rest = support.pipelined(registry, values, (name,), [label], None if graph is None else {name: graph})
    assert rest == {}
    return entries[0]


def test_an_entry_that_names_a_kind_an_earlier_contraction_or_nodes_removed_is_refused_when_the_pipeline_is_stated(
    opened,
):
    started = time.perf_counter()
    held, registry, graph, names = opened
    merged = entry(registry, "graphspec.merge", {}, None, names[1])
    quiet = [entry(registry, name, {}, None, None) for name in ("graphspec.nodes", "graphspec.derive")]
    quiet.append(entry(registry, "graphspec.neighbourhood", {"neighbourhood/around": "team"}, None, None))
    message = "the entry %s of the pipeline names the kind %s, which the entry %s before it removed"
    for removes, label in REMOVERS:
        for name, values, kind in REFUSED:
            removing = {"contract/grouping": kind + ":v0"} if label is None else {"nodes/%s/keep" % kind: False}
            removal = entry(registry, removes, removing, label, None)
            remover = removes if label is None else label
            later = entry(registry, name, values, "again" if name == "graphspec.contract" else None, None)
            named = "again" if name == "graphspec.contract" else name
            with pytest.raises(ValueError, match=re.escape(message % (named, kind, remover))):
                resolved(registry, [removal, later])
            if name == "graphspec.contract" or values.get("nodes/item/keep") is False:
                with pytest.raises(ValueError, match=re.escape(message % (remover, kind, named))):
                    resolved(registry, [later, removal])
            else:
                assert [step.name for step in resolved(registry, [later, removal])] == [named, remover]
            assert len(resolved(registry, [removal, merged, later])) == 3
            assert len(resolved(registry, [removal] + quiet)) == 4
    LOG.info("refused exit cases=%d seconds=%.3f", len(REFUSED) * len(REMOVERS), time.perf_counter() - started)


@timed("added.runs")
def runs(graph):
    begun, last, result = {}, {}, {}
    for row in np.flatnonzero(graph.fact_present).tolist():
        key, step = (int(graph.fact_code[row]), int(graph.fact_source[row]), int(graph.fact_target[row])), int(graph.fact_step[row])
        if last.get(key) not in (step, step - 1):
            begun[key] = step
        last[key] = step
        result[row] = begun[key]
    return result


@timed("added.stepped")
def stepped(graph, registry, rows, begun, step, aggregate):
    table, direction, column = step
    codes = set(np.concatenate([registry.coded[group] for group in registry.groups]).tolist()) if table == "*" else set(registry.coded[table].tolist())
    found, ages = {}, {}
    for row in rows:
        if int(graph.fact_code[row]) not in codes:
            continue
        value = 1.0
        if column is not None:
            reference = int(graph.fact_ref[row])
            value = np.nan if reference < 0 else float(graph.value_table[reference, registry.attributes(table).index(column)])
            if value != value:
                continue
        a, b = int(graph.fact_source[row]), int(graph.fact_target[row])
        ends = {"forward": [(a, b)], "backward": [(b, a)], "both": [(a, b), (b, a)]}[direction]
        for pair in ends:
            found[pair] = value if pair not in found or aggregate in (None, "paths") else PLUS[aggregate](found[pair], value)
            ages[pair] = max(ages.get(pair, -1), begun[row])
    return found, ages


PLUS = dict(paths=lambda a, b: a + b, product=lambda a, b: a + b, shortest=min, widest=max)
TIMES = dict(paths=lambda a, b: a * b, product=lambda a, b: a * b, shortest=lambda a, b: a + b, widest=min)


@timed("added.latest")
def latest(found, ages, tokens, cap, middle):
    if not cap:
        return found
    kept = set()
    for node in {pair[middle] for pair in found}:
        others = sorted(((ages[pair], int(tokens[pair[1 - middle]])), pair) for pair in found if pair[middle] == node)
        kept |= {pair for _, pair in others[::-1][:cap]}
    return {pair: value for pair, value in found.items() if pair in kept}


@timed("added.multiplied")
def multiplied(left, right, aggregate):
    found = {}
    for (a, b), x in left.items():
        for (c, d), y in right.items():
            if b == c:
                value = TIMES[aggregate or "paths"](x, y)
                found[a, d] = value if (a, d) not in found else PLUS[aggregate or "paths"](found[a, d], value)
    return found


@timed("added.enumerated")
def enumerated(graph, registry, stated):
    steps = [tuple(part.split(":")) + (None,) * (3 - len(part.split(":"))) for part in stated["product/path"].split(",")]
    valued = all(step[2] is not None for step in steps)
    aggregate = stated["product/aggregate"] if valued else ("paths" if stated["product/paths"] else None)
    begun, tokens, result = runs(graph), np.asarray(graph.entity_token).astype(np.uint64), {}
    for step in range(graph.steps):
        rows = np.flatnonzero(graph.fact_present & (graph.fact_step == step)).tolist()
        held = [stepped(graph, registry, rows, begun, path_step, aggregate) for path_step in steps]
        if len(held) == 2:
            first = latest(*held[0], tokens, stated.get("product/cap", 0), 1)
            power = multiplied(first, latest(*held[1], tokens, stated.get("product/cap", 0), 0), aggregate)
        else:
            power = held[0][0]
        power = {pair: value for pair, value in power.items() if pair[0] != pair[1]}
        total = {pair: (value, 1) for pair, value in power.items()}
        walked = power
        for distance in range(2, stated.get("product/length", 1) + 1):
            walked = multiplied(walked, power, aggregate)
            for pair, value in walked.items():
                if pair in total:
                    total[pair] = (PLUS[aggregate or "paths"](total[pair][0], value), total[pair][1])
                else:
                    total[pair] = (value, distance)
        symmetric = [(table, {"forward": "backward", "backward": "forward"}.get(way, way), column) for table, way, column in steps[::-1]] == steps
        for (a, b), (value, distance) in total.items():
            if a == b:
                continue
            ends = tuple(sorted((int(tokens[a]), int(tokens[b])))) if symmetric else (int(tokens[a]), int(tokens[b]))
            held = (() if aggregate is None else (np.float32(value),)) + ((np.float32(distance),) if stated.get("product/length", 1) > 1 else ())
            result[(int(graph.step_time[step]),) + ends] = held
    return result, symmetric


@timed("added.multiplied_facts")
def multiplied_facts(found, edge):
    tokens, present, absent = np.asarray(found.entity_token).astype(np.uint64), {}, set()
    for row in np.flatnonzero(found.fact_code == edge.code).tolist():
        ends = (int(tokens[found.fact_source[row]]), int(tokens[found.fact_target[row]]))
        key = (int(found.step_time[found.fact_step[row]]),) + (tuple(sorted(ends)) if edge.symmetric else ends)
        reference = int(found.fact_ref[row])
        if found.fact_present[row]:
            values = found.value_table[reference, : len(edge.attributes)] if reference >= 0 else ()
            present[key] = tuple(np.float32(value) for value in values)
        else:
            absent.add(key)
    return present, absent


PRODUCTS = (
    {"product/path": "visits:forward,visits:backward", "product/length": 1, "product/paths": True, "product/cap": 2, "product/encoder": "standard"},
    {"product/path": "*:both", "product/length": 3, "product/paths": False, "product/encoder": "standard"},
    {"product/path": "visits:forward:weight,visits:backward:weight", "product/length": 2, "product/aggregate": "product", "product/cap": 0, "product/encoder": "standard"},
    {"product/path": "visits:both:weight", "product/length": 2, "product/aggregate": "widest", "product/encoder": "standard"},
    {"product/path": "near:forward", "product/length": 3, "product/paths": True, "product/encoder": "standard"},
    {"product/path": "visits:forward,located:backward", "product/paths": False, "product/cap": 1},
    {"product/path": "visits:forward:weight,located:backward:distance", "product/aggregate": "shortest", "product/cap": 0, "product/encoder": "logarithm"},
)


def test_a_product_of_relations_and_its_closure_equal_the_walks_enumerated_through_the_relations_it_names(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    found = {}
    for stated in PRODUCTS:
        steps = resolved(registry, [dict(operation="graphspec.product", parameters=stated)])
        grown = pipeline.grown(registry, steps, None)
        derived = segments.whole(graph, {}, grown, steps, SEED, LIMITS, None)
        expected, symmetric = enumerated(graph, registry, stated)
        edge = grown.by_name["product"]
        assert edge.symmetric == symmetric and len(expected), stated["product/path"]
        present, absent = multiplied_facts(derived, edge)
        assert present.keys() == expected.keys(), stated["product/path"]
        assert all(np.allclose(present[key], expected[key], rtol=1e-6) for key in expected), stated["product/path"]
        times = [int(moment) for moment in graph.step_time]
        ended = {(times[times.index(key[0]) + 1],) + key[1:] for key in expected if key[0] != times[-1]}
        assert absent == {key for key in ended if key not in expected}, stated["product/path"]
        found[stated["product/path"]] = present
    uncapped = enumerated(graph, registry, dict(PRODUCTS[0], **{"product/cap": 0}))[0]
    assert set(found["visits:forward,visits:backward"]) < set(uncapped)
    projected = {"derive/project:visits:sources:person:person": True, "derive/project_cap": 2, "derive/hops": 3}
    steps = resolved(registry, support.pipelined(registry, projected, ("graphspec.derive",))[0])
    derived = segments.whole(graph, {}, registry, steps, SEED, LIMITS, None)
    pairs = multiplied_facts(derived, registry.by_name["project:visits:sources:person:person"])[0]
    assert pairs.keys() == found["visits:forward,visits:backward"].keys()
    for distance in (2, 3):
        hops = multiplied_facts(derived, registry.by_name["hop:%d" % distance])[0]
        assert hops.keys() == {key for key, value in found["*:both"].items() if value == (distance,)} != set()
    bare = support.registry("bare", ["thing"], [])
    assert OPERATIONS.named("graphspec.product").stated(bare) == ()
    with pytest.raises(ValueError, match="multiplies the relations of a bank, and the bank bare states none"):
        pipeline.grown(bare, resolved(bare, [dict(operation="graphspec.product", parameters={})]), None)
    LOG.info("product exit paths=%d seconds=%.3f", len(found), time.perf_counter() - started)
