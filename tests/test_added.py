import logging
import time

import numpy as np
import pytest
import support
from registered import KEPT, LINKED, RUNNING

from conftest import SPLITS
from graphspec import REPRESENTATIONS, Representation
from graphspec.budgets import GraphLimits
from graphspec.engine import segments
from graphspec.engine.edits import sliced
from graphspec.engine.transform import SEED
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.operations.nearby import NEARBY, fillable, measured, placed
from graphspec.operations.pipeline import resolved
from graphspec.operations.rows import keyed_relations
from graphspec.representations.registry import FAMILIES, families
from tuning.bank.bank import Deriver
from tuning.families.model_spec import DERIVES, derived
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("added")
SHIPPED = dict(structure=("coupling", "events", "mixed"), sample=("neighbors", "walks"), encode=())
ADDED = (LINKED, KEPT, RUNNING)
LIMITS = GraphLimits("cpu")


@pytest.fixture(scope="module")
def opened(bank):
    held = Deriver(bank, "cpu", RESIDENT)
    names = [graph["name"] for graph in held.manifest["graphs"]]
    return held, support.fitted(held), support.shown(held, 0).dense(), names


def observed(graph):
    tokens, table = np.asarray(graph.entity_token), np.asarray(graph.x_table)
    return {
        (int(graph.step_time[step]), int(tokens[entity])): table[reference].tobytes()
        for step, entity, reference in zip(graph.obs_step.tolist(), graph.obs_entity.tolist(), graph.obs_ref.tolist())
    }


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


def test_nearby_joins_the_rows_whose_columns_lie_within_a_distance_or_are_nearest(opened):
    started = time.perf_counter()
    held, registry, graph, names = opened
    edge, place = registry.by_name["near"], registry.kind("place")
    assert fillable(registry) == ("owns", "member", "near")
    assert measured(registry, "place", "place") == ("v0", "v1", "v2", "v3")
    values, flags = placed(registry, measured(registry, "place", "place"), "near", "place", "place")
    table, tokens = np.asarray(graph.x_table), np.asarray(graph.entity_token)
    for rule, stated_rule, count in (("threshold", {"nearby/threshold": 1.5}, None), ("nearest", {"nearby/k": 3}, 3)):
        entry = dict(
            operation="graphspec.nearby", parameters={"nearby/relation": "near", "nearby/rule": rule, **stated_rule}
        )
        found = segments.whole(graph, {}, registry, resolved(registry, [entry]), SEED, LIMITS, None)
        expected = set()
        for step in range(graph.steps):
            rows = np.flatnonzero(
                (graph.obs_step == step) & (graph.kinds[graph.obs_entity] == place) & ~graph.obs_anchor
            )
            rows = rows[np.all(table[graph.obs_ref[rows]][:, flags] > 0, axis=1)]
            vectors = table[graph.obs_ref[rows]][:, values].astype(np.float64)
            who = tokens[graph.obs_entity[rows]].astype(np.uint64)
            for first in range(len(rows)):
                distance = np.sqrt(((vectors - vectors[first]) ** 2).sum(1))
                others = sorted((distance[second], int(who[second])) for second in range(len(rows)) if second != first)
                chosen = others[:count] if rule == "nearest" else [held for held in others if held[0] <= 1.5]
                expected |= {(int(graph.step_time[step]), int(who[first]), token) for _, token in chosen}
        new, before = added(found, graph, edge.code)
        assert new == sorted(expected - before) and new
    plain = support.registry("plain", ["thing"], [])
    carry, dense = segments.opening(graph, {})
    assert NEARBY.stated(plain) == ()
    with pytest.raises(ValueError, match="fills a relation stated without columns between kinds of node that share"):
        NEARBY(dense, plain, {}, SEED, LIMITS, carry, NEARBY.initial(plain, {}), None)
    LOG.info("nearby exit seconds=%.3f", time.perf_counter() - started)
