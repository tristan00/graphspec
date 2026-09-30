import gc
import logging
import time

import numpy as np
import pytest
import support
from registered import RUNNING, TALLY

from conftest import SPLITS
from graphspec import OPERATIONS, REPRESENTATIONS
from graphspec.budgets import GraphLimits
from graphspec.engine import join, schedule, segments, transform
from graphspec.engine.edits import Edit, edited
from graphspec.engine.segments import Derivation
from graphspec.engine.stitch import Appender
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import Arrays
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.views.graph import DENSE, Graph
from tuning.bank.bank import Deriver

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
    "relations/types": "untyped",
    "relations/parallel": "merge",
}
SETTINGS = {"coupling/periods": 1, "events/periods": 1, "mixed/trace": 2}
GROUPS = specs.GROUPS + (RUNNING.name,)
REPEATS, MARGIN = 7, 1.1


@pytest.fixture
def registered():
    OPERATIONS.register(TALLY.name, TALLY)
    REPRESENTATIONS.register(RUNNING.name, RUNNING)
    yield
    del OPERATIONS.held[TALLY.name]
    del REPRESENTATIONS.held[RUNNING.name]


def every(registry, joined):
    entries, rest = support.pipelined(registry, VALUES, support.SEVEN + (TALLY.name,))
    assert rest == {}
    graph = dict(SETTINGS, **({"graphs/join": "single", "graphs/join_size": 4} if joined else {}))
    return pipeline.resolved(registry, entries), specs.configured(specs.settings(registry, specs.GROUPS), graph)


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
        elif name in ("period", "step_graph"):
            arrays[name] = np.asarray(arrays[name])[:stop]
    return type(graph)(graph.name, graph.keys, stop, **arrays)


@GROUP
@pytest.mark.parametrize("draw", range(4))
def test_a_joined_group_derived_in_segments_at_any_times_equals_its_physical_join(bank, draw):
    opened = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(opened)
    rng = np.random.default_rng(draw)
    template = [dict(operation=name) for name in support.SEVEN]
    stated = specs.draw(
        pipeline.space(registry, template),
        lambda parameter: True if parameter.name.count("/") == 1 else parameter.uniform(rng),
    )
    steps = pipeline.resolved(registry, pipeline.entries(template, stated))
    spec = specs.draw(
        specs.settings(registry, specs.GROUPS),
        lambda parameter: "single" if parameter.name == "graphs/join" else parameter.uniform(rng),
    )
    graphs = [support.shown(opened, index).dense() for index in range(4)]
    graphs = [shortened(graph, int(rng.integers(1, graph.steps + 1))) for graph in graphs]
    physical = join.joined(graphs, registry, spec)
    whole = dict(spec, **{"graphs/join": "separate"})
    limits = GraphLimits("cpu")
    reference = Derivation.begun([physical], registry, steps, whole, specs.GROUPS, transform.SEED, limits, None, 0)
    reference = reference.continued(physical.steps, None)
    line = Derivation.begun(graphs, registry, steps, spec, specs.GROUPS, transform.SEED, limits, None, 0)
    for high in sorted(set(rng.integers(1, physical.steps, 3).tolist())) + [physical.steps]:
        line = line.continued(high, None)
    derived, expected = line.derived(), reference.derived()
    assert [name for name in DENSE if not identical(derived.arrays[name], expected.arrays[name])] == []
    found, held = line.arrays(), reference.arrays()
    assert sorted(found) == sorted(held)
    assert [name for name in held if not identical(found[name], held[name])] == []


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
            for loaded in (np.load(folder / (name + ".npy"), mmap_mode="r"), closed[name]):
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


def begun(graph, registry, steps, spec, room):
    return Derivation.begun([graph], registry, steps, spec, GROUPS, transform.SEED, LIMITS, None, room)


@GROUP
def test_a_fork_its_sibling_forks_and_the_trunk_equal_the_whole_graphs_derived_separately(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
    registry, graph = support.fitted(opened), support.shown(opened, 0).dense()
    steps, spec = every(registry, False)
    assert sorted(step.operation.name for step in steps) == sorted(OPERATIONS.names())
    assert sorted(GROUPS) == sorted(REPRESENTATIONS.names())
    for step in steps:
        alone = segments.whole(graph, registry, (step,), LIMITS, None)
        assert [name for name in DENSE if not identical(alone.arrays[name], graph.arrays[name])], step.name
    moment = graph.steps - 1
    changes = edits(graph, moment)
    root = begun(graph, registry, steps, spec, 1)
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
        expected = begun(whole, registry, steps, spec, 0).continued(graph.steps, None)
        derived, reference = derivation.derived(), expected.derived()
        assert [name for name in DENSE if not identical(derived.arrays[name], reference.arrays[name])] == []
        found, held = derivation.arrays(), expected.arrays()
        assert sorted(found) == sorted(held) and "running" in found
        assert [name for name in held if not identical(found[name], held[name])] == []
    with pytest.raises(ValueError, match="and it was begun with room for 0"):
        begun(graph, registry, steps, spec, 0).continued(graph.steps, changes[0])
    missing = made(graph, removed=dict(observations=dict(entity=[len(graph.kinds)], step=[moment])))
    with pytest.raises(ValueError, match="which the graph does not hold"):
        history.continued(graph.steps, missing)
    joined, stated = every(registry, True)
    both = Derivation.begun([graph, graph], registry, joined, stated, GROUPS, transform.SEED, LIMITS, None, 0)
    with pytest.raises(ValueError, match="an edit applies to the steps of one graph"):
        both.continued(graph.steps + 1, changes[1])
    LOG.info("forks equal whole graphs exit seconds=%.3f", time.perf_counter() - started)


def clocked(make):
    gc.disable()
    try:
        clock = time.perf_counter()
        result = make()
        return time.perf_counter() - clock, result
    finally:
        gc.enable()


@GROUP
def test_n_edits_at_one_time_take_one_derivation_of_the_history_and_n_of_the_last_step(bank, registered):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
    registry, graph = support.fitted(opened), support.shown(opened, 1).dense()
    steps, spec = every(registry, False)
    moment = graph.steps - 1
    changes = edits(graph, moment)
    found = dict(history=[], last=[], fork=[])
    for _ in range(REPEATS):
        spent, held = clocked(lambda: begun(graph, registry, steps, spec, 1).continued(moment, None))
        found["history"].append(spent)
        forks = ()
        for change in changes:
            found["last"].append(clocked(lambda: held.continued(graph.steps, None))[0])
            spent, fork = clocked(lambda: held.forked(graph.steps, (change,)))
            found["fork"].append(spent)
            forks += fork
    history, last, fork = (float(np.median(found[name])) for name in ("history", "last", "fork"))
    derived = [item for derivation in (held,) + forks for item in derivation.segments]
    assert [item.high - item.low for item in derived] == [moment] + [graph.steps - moment] * len(changes)
    assert all(fork.parent is held and fork.derivations()[:1] == held.derivations() for fork in forks)
    ratio = (history + len(changes) * fork) / (history + len(changes) * last)
    LOG.info(
        "edits at one time exit edits=%d history=%.4f last=%.4f fork=%.4f ratio=%.3f seconds=%.3f",
        len(changes),
        history,
        last,
        fork,
        ratio,
        time.perf_counter() - started,
    )
    assert ratio <= MARGIN, (history, last, fork)


@pytest.mark.xdist_group("workers")
def test_the_arrays_of_a_joined_group_are_equal_for_one_and_two_workers(bank, registered, tmp_path):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(opened)
    cfg = support.configuration(opened.registry, dict(view="snapshot"), support.several(opened.registry), 5, SPLITS)
    paths = [opened.view(index, cfg) for index in range(4)]
    steps, spec = every(registry, True)
    folders = {name: tmp_path / name for name in ("one", "two", "spill")}
    for folder in folders.values():
        folder.mkdir()
    members = [Graph.load(path, mapped=True) for path in paths]
    segments.variant(members, registry, steps, spec, GROUPS, transform.SEED, LIMITS, None, folders["one"])
    job = dict(members=members, paths=paths, spill=folders["spill"], destination=folders["two"])
    schedule.variants([job], registry, steps, spec, GROUPS, transform.SEED, 2, LIMITS, None)
    one, two = Arrays(folders["one"]), Arrays(folders["two"])
    assert sorted(one) == sorted(two) and "running" in one
    assert [name for name in one if not identical(one[name], two[name])] == []
    LOG.info("one and two workers exit seconds=%.3f", time.perf_counter() - started)
