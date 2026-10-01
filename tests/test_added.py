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
from graphspec.infra.timing import timed
from graphspec.operations.consecutive import CONSECUTIVE
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
    carry, dense = segments.opening(graph, {})
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
