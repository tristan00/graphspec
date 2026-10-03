import gc
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import support
from registered import DRAWING, HUB, RUNNING, TALLY

from conftest import SPLITS
from graphspec import OPERATIONS, REPRESENTATIONS
from graphspec.budgets import GraphLimits
from graphspec.engine import join, schedule, segments
from graphspec.engine.edits import Edit, edited, sliced
from graphspec.engine.segments import Derivation
from graphspec.engine.stitch import Appender
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import Arrays
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.contract import groupable
from graphspec.operations.operation import Refused
from graphspec.operations.state import EMPTY
from graphspec.views.graph import DENSE, PER_STEP, Graph, verify_dense
from tuning.bank.bank import Deriver
from tuning.searches.selector import random_suggest

LOG = logging.getLogger(__name__)
GROUP = pytest.mark.xdist_group("segments")
LIMITS = GraphLimits("cpu")
VALUES = {
    "time/periods": "every",
    "time/period_length": 2,
    "time/scheme": "window",
    "time/unit": "step",
    "time/window": 2,
    "nodes/team/filter": "total_degree",
    "nodes/team/min_degree": 1,
    "nodes/item/filter": "activity",
    "nodes/item/min_activity": 2,
    "nodes/place/filter": "sample",
    "nodes/place/sample": 0.7,
    "derive/project:visits:targets:place:place": True,
    "derive/project_cap": 8,
    "derive/complete/team": True,
    "derive/hops": 2,
    "derive/rewiring": "sdrf",
    "derive/rewiring_budget": 2,
    "relations/owns/keep": False,
    "edges/visits/persistence": 2,
    "edges/near/cap": 2,
    "edges/near/cap_order": "random",
    "edges/hop:2/drop": 0.3,
    "relation_forms/types": "typed",
    "relation_forms/parallel": "sum",
    "relation_forms/reverse": True,
    "relation_forms/loops": True,
    "merge/prefer": "second",
    "difference/match": "keys",
    "nearby/kinds": "place:place",
    "nearby/rule": "nearest",
    "nearby/k": 2,
    "nearby/encoder": "standard",
    "consecutive/pairing": "near:visits:targets",
    "consecutive/gap": 3.0,
    "neighbourhood/hops": 1,
    "neighbourhood/around": "team",
    "measures/degree": True,
    "measures/components": True,
    "measures/cores": True,
    "measures/encoder": "standard",
    "contract/grouping": "item:v0",
    "contract/aggregate": "mean",
    "contract/parallel": "sum",
    "product/path": "visits:forward,visits:backward",
    "product/length": 2,
    "product/paths": True,
    "product/cap": 2,
    "product/encoder": "standard",
    "negatives/of": "link:visits",
    "negatives/rule": "hops",
    "negatives/count": 2,
    "negatives/hops/within": 2,
}
SETTINGS = {"coupling/periods": 1, "events/periods": 1, "mixed/trace": 2}
GROUPS = specs.GROUPS + (RUNNING.name,)
ADDED = support.ADDED
TESTED = (TALLY, HUB)
SECONDS = {"graphspec.merge": 1, "graphspec.difference": 0}
ROUNDS, MARGIN = (8, 32), 1.1
SHOWN, STEPS, BARE, HOLLOW, UNVALUED = 2, 5, 1, 2, 3
EMPTIED = dict(
    VALUES,
    **{
        "difference/match": "values",
        "edges/visits/persistence": 1,
        "edges/visits/cap": 2,
        "edges/visits/cap_order": "largest:weight",
        "edges/owns/persistence": 2,
    },
)
DRAWN, SPAN, CUTS, REWIRES, JOINING, MEMBERS = 20, 8, 2, 4, 4, 4
REWIRING = "pipeline/graphspec.derive/derive/rewiring_budget"
JOINED = {"graphs/join": "single", "graphs/join_size": MEMBERS}
CONTRACTED = {"contract/grouping": "item:v0", "contract/aggregate": "mean", "contract/parallel": "keep"}
WINDOWED = {"time/scheme": "window", "time/unit": "step", "time/window": 2}
ELSEWHERE = 0x5A5A5A5A5A5A5A5A
GROUPING = "pipeline/graphspec.contract/contract/grouping"
CONTRACTING = {"pipeline/graphspec.contract": True}
UNSPLIT = "pipeline/graphspec.negatives/negatives/of"
REFUSAL = re.compile(r"the entry (\S+) of the pipeline names the kind (\w+), which the entry (graphspec\.\w+) before ")
KEPT = "pipeline/graphspec.nodes/nodes/%s/keep"
CONTRACTIONS = (
    (("graphspec.contract", CONTRACTED, None),),
    (("graphspec.window", WINDOWED, None), ("graphspec.contract", CONTRACTED, None)),
    (("graphspec.merge", {"merge/prefer": "second"}, 1), ("graphspec.contract", CONTRACTED, None)),
    (("graphspec.difference", {"difference/match": "keys"}, 0),),
)
PROPOSED = (5, 23)
UNSCALED = dict(mean=None, sd=None)
DENSE_READ = dict(obs_ref="x_table", fact_ref="value_table")
VARIANT_READ = dict(x_ref="x_table")


@pytest.fixture
def registered():
    for entry in TESTED:
        OPERATIONS.register(entry.name, entry)
    REPRESENTATIONS.register(RUNNING.name, RUNNING)
    yield
    for entry in TESTED:
        del OPERATIONS.held[entry.name]
    del REPRESENTATIONS.held[RUNNING.name]


def every(opened, registry, joined, values, extra):
    graphs = {operation: opened.manifest["graphs"][index]["name"] for operation, index in SECONDS.items()}
    assert sorted(graphs) == sorted(name for name in ADDED if OPERATIONS.named(name).graphs == 2)
    names = support.SEVEN + tuple(entry.name for entry in TESTED) + ADDED + extra
    entries, rest = support.pipelined(registry, values, names, None, graphs)
    assert rest == {}
    graph = dict(SETTINGS, **({"graphs/join": "single", "graphs/join_size": 4} if joined else {}))
    steps = pipeline.resolved(registry, entries)
    spec = specs.configured(specs.settings(registry, specs.GROUPS), graph)
    return pipeline.grown(registry, steps, None), steps, spec


@timed("segments.read")
def read(arrays, pairs):
    found = {name: np.asarray(value) for name, value in arrays.items()}
    for reference, table in pairs.items():
        rows, values = found.pop(reference), found.pop(table)
        held = np.full((len(rows),) + values.shape[1:], np.nan, values.dtype)
        held[rows >= 0] = values[rows[rows >= 0]]
        found[reference] = held
    return found


@timed("segments.unequal")
def unequal(left, right, pairs):
    left, right = read(left, pairs), read(right, pairs)
    return sorted(set(left) ^ set(right) | {name for name in set(left) & set(right) if not identical(left[name], right[name])})


def seconds(opened, steps, indices):
    started = time.perf_counter()
    result = {
        name: [support.beside(opened, index, name) for index in indices]
        for name in sorted({step.graph for step in steps if step.graph is not None})
    }
    LOG.debug("seconds exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def identical(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def shortened(graph, stop):
    arrays = dict(graph.arrays)
    tasks = int((np.asarray(graph.task_step) < stop).sum())
    kept = dict(
        obs=np.asarray(graph.obs_step) < stop,
        fact=np.asarray(graph.fact_step) < stop,
        task=np.asarray(graph.task_step) < stop,
        member=np.asarray(graph.member_task) < tasks,
        label=np.asarray(graph.label_task) < tasks,
    )
    for name in arrays:
        group = name.split("_", 1)[0]
        if group in kept:
            arrays[name] = np.asarray(arrays[name])[kept[group]]
        elif name in PER_STEP:
            arrays[name] = np.asarray(arrays[name])[:stop]
    return type(graph)(graph.name, graph.keys, stop, **arrays)


@GROUP
@pytest.mark.parametrize("draw", range(4))
def test_a_joined_group_derived_in_segments_at_any_times_equals_its_physical_join(bank, draw):
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = support.fitted(opened)
    rng = np.random.default_rng(draw)
    template = [dict(operation=name) for name in support.SEVEN]
    stated = pipeline.draw(
        registry,
        template,
        pipeline.space(registry, template),
        lambda parameter: parameter.options[-1] if parameter.name.count("/") == 1 else parameter.uniform(rng),
        {},
    )
    steps = pipeline.resolved(registry, pipeline.entries(template, stated))
    grown = pipeline.grown(registry, steps, None)
    spec = specs.draw(
        specs.settings(registry, specs.GROUPS),
        lambda parameter: "single" if parameter.name == "graphs/join" else parameter.uniform(rng),
    )
    graphs = [support.shown(opened, index).dense() for index in range(4)]
    graphs = [shortened(graph, int(rng.integers(1, graph.steps + 1))) for graph in graphs]
    physical = join.joined(graphs, registry, spec, support.SEED)
    whole = dict(spec, **{"graphs/join": "separate"})
    limits = GraphLimits("cpu")
    reference = Derivation.begun([physical], {}, grown, steps, whole, specs.GROUPS, support.SEED, limits, None, 0)
    reference = reference.continued(physical.steps, None)
    line = Derivation.begun(graphs, {}, grown, steps, spec, specs.GROUPS, support.SEED, limits, None, 0)
    for high in sorted(set(rng.integers(1, physical.steps, 3).tolist())) + [physical.steps]:
        line = line.continued(high, None)
    derived, expected = line.derived(), reference.derived()
    assert unequal(derived.arrays, expected.arrays, DENSE_READ) == []
    found, held = line.arrays(), reference.arrays()
    assert sorted(found) == sorted(held)
    assert unequal(found, held, VARIANT_READ) == []


@timed("segments.renamed")
def renamed(second, registry):
    tokens = np.asarray(second.entity_token).copy()
    chosen = (np.asarray(second.kinds) == registry.kind("person")) & (np.arange(len(tokens)) % 2 == 0)
    tokens[chosen] ^= np.uint64(ELSEWHERE)
    return second.replace(entity_token=tokens)


@GROUP
def test_contraction_merge_and_difference_in_a_joined_group_equal_the_physical_join_of_its_graphs(bank):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = support.fitted(opened)
    graphs = [support.shown(opened, index).dense() for index in range(MEMBERS)]
    names = [graph["name"] for graph in opened.manifest["graphs"]]
    spec = specs.configured(specs.settings(registry, specs.GROUPS), dict(SETTINGS, **JOINED))
    physical = join.joined(graphs, registry, spec, support.SEED)
    bounds = np.cumsum([member.steps for member in graphs])
    cuts = sorted(set(bounds[:-1].tolist()) | set((bounds[:-1] + 1).tolist())) + [physical.steps]
    separate, counts, shared = dict(spec, **{"graphs/join": "separate"}), [], []
    for stated in CONTRACTIONS:
        entries = [
            dict(operation=name, parameters=values, **({} if second is None else dict(graph=names[second])))
            for name, values, second in stated
        ]
        steps = pipeline.resolved(registry, entries)
        grown = pipeline.grown(registry, steps, None)
        beside = {
            entry["graph"]: [renamed(support.beside(opened, index, entry["graph"]), registry) for index in range(MEMBERS)]
            for entry in entries
            if "graph" in entry
        }
        whole = {name: [join.joined(held, registry, spec, support.SEED)] for name, held in beside.items()}
        reference = Derivation.begun(
            [physical], whole, grown, steps, separate, specs.GROUPS, support.SEED, LIMITS, None, 0
        )
        reference = reference.continued(physical.steps, None)
        line = Derivation.begun(graphs, beside, grown, steps, spec, specs.GROUPS, support.SEED, LIMITS, None, 0)
        for high in cuts:
            line = line.continued(high, None)
        derived, expected = line.derived(), reference.derived()
        assert unequal(derived.arrays, expected.arrays, DENSE_READ) == [], stated
        assert unequal(line.arrays(), reference.arrays(), VARIANT_READ) == [], stated
        kinds = np.asarray(derived.kinds)
        shared.append(len(kinds) - len(physical.kinds) - int((kinds >= len(registry.declared)).sum()))
        made = kinds == (grown.kind("contract") if "contract" in grown.nodes else -1)
        counts.append(int(made.sum()))
        assert np.asarray(derived.entity_connected)[made].all()
        assert len(np.unique(np.asarray(derived.entity_token)[made])) == made.sum()
    assert counts[0] == counts[1] > 0 and counts[2] > 0 and shared[0] == shared[1] == 0 < shared[2] and shared[3] > 0
    LOG.info(
        "joined group exit groups=%s second=%s seconds=%.3f", counts, shared, time.perf_counter() - started
    )


@GROUP
def test_appender_writes_loadable_arrays_of_every_shape(tmp_path):
    pieces = [
        dict(
            flags=np.asarray([True, False]),
            wide=np.arange(6, dtype=np.float16).reshape(2, 3),
            hollow=np.zeros((2, 0, 2), np.float32),
            later=np.zeros(0, np.int64),
        ),
        dict(
            flags=np.asarray([True]),
            wide=np.arange(3, dtype=np.float16).reshape(1, 3),
            hollow=np.zeros((3, 0, 2), np.float32),
            later=np.asarray([7, 8], np.int64),
        ),
    ]
    for limit in (0, 1 << 20):
        folder = tmp_path / str(limit)
        folder.mkdir()
        appender = Appender(folder, limit)
        for piece in pieces:
            appender.add(piece)
        closed = appender.close()
        for name in pieces[0]:
            expected = np.concatenate([piece[name] for piece in pieces])
            for loaded in (Arrays(folder)[name], closed[name]):
                assert loaded.dtype == expected.dtype and loaded.shape == expected.shape
                assert np.array_equal(np.asarray(loaded), expected)
        (tmp_path / ("strict%d" % limit)).mkdir()
        strict = Appender(tmp_path / ("strict%d" % limit), limit)
        strict.add(dict(flags=np.zeros(1, bool)))
        with pytest.raises(ValueError, match="changed dtype or shape"):
            strict.add(dict(flags=np.zeros(1, np.int8)))
        strict.abandon()


def made(graph, entities=(), removed=None, added=None):
    width = graph.value_table.shape[1]
    nothing = dict(
        observations=dict(entity=[], step=[], x=np.zeros((0, graph.x_table.shape[1]))),
        facts=dict(source=[], target=[], code=[], step=[], present=[], values=np.zeros((0, width))),
    )
    return Edit(
        dict(kind=[kind for kind, _ in entities], token=np.asarray([token for _, token in entities], np.uint64)),
        dict(
            dict(observations=dict(entity=[], step=[]), facts=dict(code=[], source=[], target=[], step=[])),
            **(removed or {}),
        ),
        dict(nothing, **(added or {})),
    )


def edits(graph, moment):
    entity, table = np.asarray(graph.obs_entity), np.asarray(graph.x_table, np.float32)
    at = np.asarray(graph.obs_step) == moment
    stated = np.flatnonzero((np.asarray(graph.fact_step) == moment) & np.asarray(graph.fact_present))
    source, target, code = (int(graph.arrays["fact_" + name][stated[0]]) for name in ("source", "target", "code"))
    other = {name: [int(graph.arrays["fact_" + name][stated[-1]])] for name in ("source", "target", "code")}
    asked = np.isin(np.asarray(graph.member_task), np.flatnonzero(np.asarray(graph.task_step) == moment))
    busy = np.r_[graph.fact_source[stated], graph.fact_target[stated], graph.member_source[asked]]
    earlier = entity[np.asarray(graph.obs_step) < moment]
    quiet = int(np.setdiff1d(np.intersect1d(entity[at], earlier), np.r_[busy, graph.member_target[asked]])[0])
    x = table[np.asarray(graph.obs_ref)[at & (entity == quiet)]]
    shape = table[np.asarray(graph.obs_ref)[at & (entity == source)]]
    fresh, token = len(graph.kinds), int(np.asarray(graph.entity_token).max()) + 1
    gone = dict(observations=dict(entity=[quiet], step=[moment]))
    return (
        made(
            graph,
            [(int(graph.kinds[source]), token)],
            added=dict(
                observations=dict(entity=[fresh], step=[moment], x=shape),
                facts=dict(
                    source=[fresh],
                    target=[target],
                    code=[code],
                    step=[moment],
                    present=[True],
                    values=np.full((1, graph.value_table.shape[1]), np.nan),
                ),
            ),
        ),
        made(graph, removed=dict(gone, facts=dict(other, step=[moment]))),
        made(graph, removed=gone, added=dict(observations=dict(entity=[quiet], step=[moment], x=x + 1))),
    )


def begun(graph, beside, registry, steps, spec, room):
    return Derivation.begun([graph], beside, registry, steps, spec, GROUPS, support.SEED, LIMITS, None, room)


@GROUP
def test_a_fork_its_sibling_forks_and_the_trunk_equal_the_whole_graphs_derived_separately(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    graph = support.shown(opened, 0).dense()
    registry, steps, spec = every(opened, support.fitted(opened), False, VALUES, ())
    beside = seconds(opened, steps, [0])
    assert sorted(step.operation.name for step in steps) == sorted(OPERATIONS.names())
    assert sorted(GROUPS) == sorted(REPRESENTATIONS.names())
    widened = segments.whole(graph, {}, registry, (), support.SEED, LIMITS, None)
    for step in steps:
        given = {name: held[0] for name, held in beside.items() if name == step.graph}
        alone, drawn = segments.apart(graph, given, registry, (step,), support.SEED, LIMITS, None)
        changed = [name for name in DENSE if not identical(alone.arrays[name], widened.arrays[name])]
        assert changed or len(drawn["step"]), step.name
    moment = graph.steps - 1
    changes = edits(graph, moment)
    root = begun(graph, beside, registry, steps, spec, 1)
    history = root.continued(moment, None)
    trunk = history.continued(graph.steps, None)
    forks = history.forked(graph.steps, changes)
    assert history.parent is root and [(item.member, item.low, item.high) for item in history.segments] == [
        (0, 0, moment)
    ]
    for derivation, change in [(trunk, None)] + list(zip(forks, changes)):
        assert derivation.parent is history
        assert [(item.member, item.low, item.high) for item in derivation.segments] == [(0, moment, graph.steps)]
        assert all(mine is held for mine, held in zip(derivation.derivations(), history.derivations()))
        whole = graph if change is None else edited(graph, change)
        expected = begun(whole, beside, registry, steps, spec, 0).continued(graph.steps, None)
        derived, reference = derivation.derived(), expected.derived()
        assert unequal(derived.arrays, reference.arrays, DENSE_READ) == []
        found, held = derivation.arrays(), expected.arrays()
        assert sorted(found) == sorted(held) and "running" in found
        assert unequal(found, held, VARIANT_READ) == []
    with pytest.raises(ValueError, match="and it was begun with room for 0"):
        begun(graph, beside, registry, steps, spec, 0).continued(graph.steps, changes[0])
    missing = made(graph, removed=dict(observations=dict(entity=[len(graph.kinds)], step=[moment])))
    with pytest.raises(ValueError, match="which the graph does not hold"):
        history.continued(graph.steps, missing)
    _, joined, stated = every(opened, support.fitted(opened), True, VALUES, ())
    both = Derivation.begun(
        [graph, graph],
        {name: held * 2 for name, held in beside.items()},
        registry,
        joined,
        stated,
        GROUPS,
        support.SEED,
        LIMITS,
        None,
        0,
    )
    with pytest.raises(ValueError, match="an edit applies to the steps of one graph"):
        both.continued(graph.steps + 1, changes[1])
    LOG.info("forks equal whole graphs exit seconds=%.3f", time.perf_counter() - started)


@timed("segments.tokened")
def tokened(derived, drawn):
    tokens = np.asarray(derived.entity_token)
    return dict(drawn, source=tokens[drawn["source"]], target=tokens[drawn["target"]])


@GROUP
def test_no_operation_reads_or_offers_the_relations_negatives_add_as_candidates(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    graph, registry = support.shown(opened, 0).dense(), support.fitted(opened)
    graphs = {operation: opened.manifest["graphs"][index]["name"] for operation, index in SECONDS.items()}
    drawing = support.pipelined(registry, VALUES, ("graphspec.negatives",))[0]
    first = pipeline.resolved(registry, drawing)
    grown = pipeline.grown(registry, first, None)
    assert grown.candidates and [edge.name for edge in grown.edges] == [edge.name for edge in registry.edges]
    assert support.unoffered(registry, grown) == []
    drawn = tokened(*segments.apart(graph, {}, grown, first, support.SEED, LIMITS, None))
    assert len(drawn["step"])
    for name in sorted(set(OPERATIONS.names()) - {"graphspec.negatives"}):
        entry = support.pipelined(registry, VALUES, (name,), None, graphs)[0]
        given = {graphs[name]: support.beside(opened, 0, graphs[name])} if name in graphs else {}
        alone, after = pipeline.resolved(registry, entry), pipeline.resolved(registry, drawing + entry)
        plain = segments.whole(graph, given, pipeline.grown(registry, alone, None), alone, support.SEED, LIMITS, None)
        found, held = segments.apart(graph, given, pipeline.grown(registry, after, None), after, support.SEED, LIMITS, None)
        assert unequal(found.arrays, plain.arrays, DENSE_READ) == [], name
        assert unequal(tokened(found, held), drawn, {}) == [] and unequal(plain.arrays, graph.arrays, DENSE_READ), name
    LOG.info("candidates unread exit operations=%d seconds=%.3f", len(OPERATIONS.names()), time.perf_counter() - started)


@timed("segments.emptied")
def emptied(graph):
    graph = sliced(graph, SHOWN, SHOWN + STEPS)
    moments = np.asarray(graph.task_step)
    tasks = np.flatnonzero(moments != HOLLOW)
    number = np.full(len(moments), -1, np.int64)
    number[tasks] = np.arange(len(tasks))
    observed = np.asarray(graph.obs_step) != HOLLOW
    first = np.full(len(graph.kinds), STEPS, np.int64)
    np.minimum.at(first, np.asarray(graph.obs_entity)[observed], np.asarray(graph.obs_step)[observed])
    moment = np.asarray(graph.fact_step)
    known = np.asarray(graph.fact_present) | (np.maximum(first[graph.fact_source], first[graph.fact_target]) <= moment)
    kept = dict(
        obs=observed,
        fact=~np.isin(moment, (BARE, HOLLOW)) & known,
        task=moments != HOLLOW,
        member=np.isin(np.asarray(graph.member_task), tasks),
        label=np.isin(np.asarray(graph.label_task), tasks),
    )
    arrays = {
        name: np.asarray(value)[kept[name.split("_", 1)[0]]] if name.split("_", 1)[0] in kept else value
        for name, value in graph.arrays.items()
    }
    for name in ("member_task", "label_task"):
        arrays[name] = number[arrays[name]].astype(graph.arrays[name].dtype)
    unvalued = arrays["fact_step"] == UNVALUED
    arrays["fact_ref"] = np.where(unvalued, -1, arrays["fact_ref"]).astype(arrays["fact_ref"].dtype)
    return type(graph)(graph.name, graph.keys, graph.steps, **arrays)


@GROUP
def test_every_operation_derives_steps_that_hold_nothing_in_segments_as_it_derives_them_whole(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    graph = emptied(support.shown(opened, 0).dense())
    registry, steps, spec = every(opened, support.fitted(opened), False, EMPTIED, ())
    beside = {name: [emptied(held[0])] for name, held in seconds(opened, steps, [0]).items()}
    assert sorted(step.operation.name for step in steps) == sorted(OPERATIONS.names())
    for held in [graph] + [held[0] for held in beside.values()]:
        verify_dense(held, support.fitted(opened))
        facts, observed = np.asarray(held.fact_step), np.asarray(held.obs_step)
        assert np.isin([BARE, UNVALUED], observed).all() and HOLLOW not in observed
        assert not np.isin([BARE, HOLLOW], facts).any() and (held.fact_ref[facts == UNVALUED] < 0).all()
        assert (facts == UNVALUED).any() and (held.fact_ref[facts != UNVALUED] >= 0).all()
    unchanged = begun(graph, {}, registry, (), spec, 0).continued(graph.steps, None).derived()
    for step in steps:
        given = {name: held for name, held in beside.items() if name == step.graph}
        whole = begun(graph, given, registry, (step,), spec, 0).continued(graph.steps, None)
        line = begun(graph, given, registry, (step,), spec, 0)
        for high in range(1, graph.steps + 1):
            line = line.continued(high, None)
        derived, expected, made = line.derived(), whole.derived(), whole.arrays()
        assert unequal(expected.arrays, unchanged.arrays, DENSE_READ) or len(made["candidate_step"]), step.name
        assert unequal(derived.arrays, expected.arrays, DENSE_READ) == [], step.name
        assert unequal(line.arrays(), made, VARIANT_READ) == [], step.name
    LOG.info("steps that hold nothing exit operations=%d seconds=%.3f", len(steps), time.perf_counter() - started)


@timed("segments.proposed")
def proposed(directory, seeds):
    opened = Deriver(directory, "cpu", support.SEED, RESIDENT)
    names = [graph["name"] for graph in opened.manifest["graphs"]]
    shipped = [name for name in sorted(OPERATIONS.names()) if OPERATIONS.named(name).namespace == "graphspec"]
    found = []
    for seed in seeds:
        order = [shipped[place] for place in np.random.default_rng(seed).permutation(len(shipped))]
        template = [dict(operation=name, **({"graph": names[SECONDS[name]]} if name in SECONDS else {})) for name in order]
        cfg = dict(random_suggest(opened.registry, seed, seed, {}, 600, template)[0], splits=SPLITS)
        steps, registry, partitions = opened.stated(cfg)[1], opened.fitted(cfg), opened.partitions(cfg)
        for graph in opened.viewed(cfg).graphs:
            view = Graph.load(opened.view(graph["index"], cfg)).dense()
            beside = {name: held[0].dense() for name, held in opened.seconds(steps, [graph["index"]], cfg, False).items()}
            verify_dense(segments.whole(view, beside, registry, steps, support.SEED, LIMITS, partitions), registry)
        flat = [name for name in registry.schema.added_columns() if registry.encoding.states[name] == UNSCALED]
        found.append((seed, len(steps), flat))
    return found


@GROUP
def test_random_pipelines_and_proposals_of_every_operation_derive_on_every_shared_bank_whole_and_in_segments(
    bank, path_bank, varied, sets_bank, registered
):
    started = time.perf_counter()
    banks = dict(task=bank, path=path_bank, varied=varied, sets=sets_bank)
    proposals = {label: proposed(directory, PROPOSED) for label, directory in banks.items()}
    assert any(flat for held in proposals.values() for _, _, flat in held), proposals
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = support.fitted(opened)
    shown = [sliced(support.shown(opened, index).dense(), 0, SPAN) for index in range(MEMBERS)]
    graph = shown[0]
    graphs = {operation: opened.manifest["graphs"][index]["name"] for operation, index in SECONDS.items()}
    beside = {name: [sliced(support.beside(opened, 0, name), 0, SPAN)] for name in sorted(set(graphs.values()))}
    unsplit = ["link:%s" % table for table in registry.groups]
    bounds = {REWIRING: dict(low=1, high=REWIRES), UNSPLIT: unsplit}
    spec = specs.configured(specs.settings(registry, specs.GROUPS), SETTINGS)
    joined = specs.configured(specs.settings(registry, specs.GROUPS), dict(SETTINGS, **JOINED))
    names, found, refused = sorted(OPERATIONS.names()), [], []
    single = [name for name in names if OPERATIONS.named(name).graphs == 1]
    for draw in range(DRAWN):
        rng = np.random.default_rng(draw)
        together, forced = draw % JOINING == 0, draw == 0
        pool = single if together else names
        order = [pool[place] for place in rng.permutation(len(pool))]
        template = [dict(operation=name, **({"graph": graphs[name]} if name in graphs else {})) for name in order]
        space = specs.narrow(pipeline.space(registry, template), dict(bounds, **(CONTRACTING if forced else {})))
        steps = None
        while steps is None:
            stated = specs.draw(space, lambda held: held.uniform(rng))
            try:
                steps = pipeline.resolved(registry, pipeline.entries(template, stated))
            except ValueError as error:
                named = REFUSAL.match(str(error))
                assert named is not None, (draw, str(error))
                assert stated[pipeline.PRESENT % named[1]] and order.index(named[1]) > order.index(named[3])
                if named[3] == "graphspec.contract":
                    assert named[2] == stated[GROUPING].split(":")[0], draw
                else:
                    assert named[3] == "graphspec.nodes" and not stated[KEPT % named[2]], draw
                refused.append((forced, named[1], named[3]))
                if not forced:
                    break
        if steps is None:
            continue
        grown = pipeline.grown(registry, steps, None)
        if together:
            picked = sorted(rng.choice(MEMBERS, int(rng.integers(2, MEMBERS + 1)), replace=False).tolist())
            members = [sliced(shown[index], 0, int(rng.integers(2, SPAN + 1))) for index in picked]
            drawn = dict(joined, **{name: bool(rng.integers(2)) for name in joined if name.startswith("graphs/connect/")})
            physical = join.joined(members, registry, drawn, support.SEED)
            whole = begun(physical, {}, grown, steps, dict(drawn, **{"graphs/join": "separate"}), 0)
            whole = whole.continued(physical.steps, None)
            line = Derivation.begun(members, {}, grown, steps, drawn, GROUPS, support.SEED, LIMITS, None, 0)
            total = physical.steps
        else:
            given = {name: held for name, held in beside.items() if name in {step.graph for step in steps}}
            whole = begun(graph, given, grown, steps, spec, 0).continued(graph.steps, None)
            line, total = begun(graph, given, grown, steps, spec, 0), graph.steps
        for high in sorted(set(rng.integers(1, total, CUTS).tolist())) + [total]:
            line = line.continued(high, None)
        expected = whole.derived()
        verify_dense(expected, grown)
        assert unequal(line.derived().arrays, expected.arrays, DENSE_READ) == [], draw
        assert unequal(line.arrays(), whole.arrays(), VARIANT_READ) == [], draw
        found.append((together, tuple(step.operation.name for step in steps)))
    alone = sum(not forced for forced, _, _ in refused)
    assert len(set(found)) + alone == DRAWN and sorted({name for _, held in found for name in held}) == names
    assert found[0][0] and "graphspec.contract" in found[0][1]
    assert {remover for _, _, remover in refused} == {"graphspec.contract", "graphspec.nodes"}
    contracting = pipeline.space(registry, [dict(operation="graphspec.contract")])
    (offered,) = [held.options for held in contracting if held.name == GROUPING]
    lacking, found = registry.extras["unvalued"], {}
    assert sorted(set(groupable(registry)) - set(offered)) == sorted(lacking) != []
    for entry in opened.manifest["graphs"]:
        truth = Facts.load(opened.truth(entry["index"]))
        kinds, entity = np.asarray(truth.kinds), np.asarray(truth.obs_entity)
        for grouping in groupable(registry):
            kind, column = grouping.split(":")
            values = np.asarray(truth.obs_values)[:, registry.encoding.positions([column])[0]]
            missing = np.unique(entity[(kinds[entity] == registry.kind(kind)) & np.isnan(values)])
            found[grouping] = found.get(grouping, 0) + len(missing)
    assert {grouping: count for grouping, count in found.items() if count and "item:" not in grouping} == lacking
    assert {grouping.split(":")[0] for grouping in offered} > {"item"}
    grouping = sorted(lacking)[0]
    kind, column = grouping.split(":")
    message = "states contract/grouping %s, which needs a value of %s on every node of %s that a reference can name, "
    message += "and %d nodes of %s in the bank %s lack one in some fact"
    with pytest.raises(Refused, match=re.escape(message % (grouping, column, kind, lacking[grouping], kind, registry.name))):
        pipeline.resolved(
            registry, [dict(operation="graphspec.contract", parameters=dict(CONTRACTED, **{"contract/grouping": grouping}))]
        )
    LOG.info(
        "random pipelines exit pipelines=%d refused=%s proposals=%s seconds=%.3f",
        len(found),
        refused,
        proposals,
        time.perf_counter() - started,
    )


def clocked(make):
    gc.disable()
    try:
        clock = time.perf_counter()
        result = make()
        return time.perf_counter() - clock, result
    finally:
        gc.enable()


def sampled(found, rounds, history, end, changes):
    started = time.perf_counter()
    for repeat in range(rounds):
        if repeat % 2 == 0:
            spent, held = clocked(history)
            found["history"].append(spent)
        forks = ()
        for change in changes:
            if repeat % 2:
                spent, made = clocked(lambda: held.forked(end, (change,)))
                trunk = clocked(lambda: held.continued(end, None))[0]
            else:
                trunk = clocked(lambda: held.continued(end, None))[0]
                spent, made = clocked(lambda: held.forked(end, (change,)))
            found["last"].append(trunk)
            found["fork"].append(spent)
            forks += made
    LOG.debug("sampled exit rounds=%d seconds=%.4f", rounds, time.perf_counter() - started)
    return held, forks


def estimated(found, count):
    started = time.perf_counter()
    history = min(found["history"])
    last, fork = (float(np.reshape(found[name], (-1, count)).min(0).sum()) for name in ("last", "fork"))
    LOG.debug("estimated exit samples=%d seconds=%.4f", len(found["last"]), time.perf_counter() - started)
    return history, last, fork, (history + fork) / (history + last)


@GROUP
def test_n_edits_at_one_time_take_one_derivation_of_the_history_and_n_of_the_last_step(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    graph = support.shown(opened, 1).dense()
    registry, steps, spec = every(opened, support.fitted(opened), False, VALUES, ())
    beside = seconds(opened, steps, [1])
    moment = graph.steps - 1
    changes = edits(graph, moment)
    found, ratio = dict(history=[], last=[], fork=[]), np.inf
    for rounds in ROUNDS:
        if ratio > MARGIN:
            held, forks = sampled(
                found,
                rounds,
                lambda: begun(graph, beside, registry, steps, spec, 1).continued(moment, None),
                graph.steps,
                changes,
            )
            history, last, fork, ratio = estimated(found, len(changes))
    derived = [item for derivation in (held,) + forks for item in derivation.segments]
    assert [item.high - item.low for item in derived] == [moment] + [graph.steps - moment] * len(changes)
    assert all(made.parent is held and made.derivations()[:1] == held.derivations() for made in forks)
    LOG.info(
        "edits at one time exit edits=%d rounds=%d history=%.4f last=%.4f fork=%.4f ratio=%.3f seconds=%.3f",
        len(changes),
        len(found["last"]) // len(changes),
        history,
        last,
        fork,
        ratio,
        time.perf_counter() - started,
    )
    assert ratio <= MARGIN, (history, last, fork)


@pytest.fixture
def drawing():
    started = time.perf_counter()
    OPERATIONS.register(DRAWING.name, DRAWING)
    yield
    del OPERATIONS.held[DRAWING.name]
    LOG.debug("drawing exit seconds=%.4f", time.perf_counter() - started)


@GROUP
def test_a_part_that_changes_its_segment_is_refused_in_one_process_and_by_the_stage_a_worker_runs(bank, drawing):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = support.fitted(opened)
    steps = pipeline.resolved(registry, [dict(operation=DRAWING.name, parameters={"test/widen": True})])
    grown = pipeline.grown(registry, steps, None)
    spec = specs.configured(specs.settings(registry, specs.GROUPS), dict(SETTINGS, **{"graphs/join": "single"}))
    members = [support.shown(opened, 0).dense(), renamed(support.shown(opened, 1).dense(), registry)]
    line = Derivation.begun(members, {}, grown, steps, spec, specs.GROUPS, support.SEED, LIMITS, None, 0)
    message = "the part first of the phase draw of the operation test.drawing changes its entities and what it knows "
    message += "of its entities of its segment; a part changes only its own state and the candidates it draws"
    with pytest.raises(ValueError, match=re.escape(message)):
        line.continued(members[0].steps, None)
    plan = line.source.plan
    carry, graph, _, _ = segments.opened(plan, 0, members[0], 0, members[0].steps, None, plan.entities, EMPTY)
    with pytest.raises(ValueError, match=re.escape(message)):
        steps[0].phase(0, graph, grown, support.SEED, LIMITS, carry, None, None, "first", None)
    LOG.info("part refused exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.xdist_group("workers")
def test_the_arrays_of_a_joined_group_are_equal_in_one_process_and_on_one_and_two_workers(
    bank, registered, drawing, tmp_path
):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), support.several(opened.registry), 5, SPLITS)
    renamed(Graph.load(opened.view(1, cfg)), opened.registry).save(tmp_path / "renamed")
    paths = [opened.view(0, cfg), tmp_path / "renamed"]
    registry, steps, spec = every(opened, support.fitted(opened), True, VALUES, (DRAWING.name,))
    seconds = opened.seconds(steps, [0, 1], cfg, True)
    folders = {name: tmp_path / str(name) for name in ("process", 1, 2, "spill1", "spill2")}
    for folder in folders.values():
        folder.mkdir()
    jobs = {
        workers: dict(
            members=[Graph.load(path, mapped=True) for path in paths],
            beside=seconds,
            paths=paths,
            spill=folders["spill%d" % workers],
            destination=folders[workers],
        )
        for workers in (1, 2)
    }
    with ThreadPoolExecutor(2) as aside:
        found = [
            aside.submit(schedule.variants, [jobs[workers]], registry, steps, spec, GROUPS, support.SEED, workers, LIMITS, None)
            for workers in (1, 2)
        ]
        members = [Graph.load(path, mapped=True) for path in paths]
        segments.variant(
            members, seconds, registry, steps, spec, GROUPS, support.SEED, LIMITS, None, folders["process"]
        )
        for future in found:
            future.result()
    held = Arrays(folders["process"])
    codes = sorted(registry.by_candidate["drawing." + name].code for name in ("parts", "widened"))
    drawn = sorted(set(np.asarray(held["candidate_code"]).tolist()))
    assert sorted(set(drawn) & set(codes)) == codes and "running" in held, (drawn, codes)
    for workers in (1, 2):
        made = Arrays(folders[workers])
        assert sorted(made) == sorted(held)
        assert [name for name in held if not identical(held[name], made[name])] == [], workers
    LOG.info("one process and one and two workers exit seconds=%.3f", time.perf_counter() - started)
