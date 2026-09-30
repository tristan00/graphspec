import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest
import support
import torch
from keyed import Keys, Spaces, dense_keyed, differing, labelled, rows, through, variant_part

from conftest import SPLITS
from graphspec.budgets import GraphLimits
from graphspec.engine import transform
from graphspec.engine.segments import Derivation
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.rows import FACTS
from graphspec.operations.state import ENTITIES
from graphspec.views.graph import Dense
from tuning.bank.bank import Deriver, grouped
from tuning.batches import inputs

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("draws")
GROUPS = ("coupling", "events", "neighbors", "walks")
SAMPLED, PAIRED = 2, 8
LIMITS = GraphLimits("cpu")
DRAWN = {
    "time/periods": "every",
    "time/period_length": 3,
    "nodes/place/filter": "sample",
    "nodes/place/sample": 0.6,
    "nodes/item/filter": "sample",
    "nodes/item/sample": 0.7,
    "derive/hops": 2,
    "derive/rewiring": "sdrf",
    "derive/rewiring_budget": 3,
    "edges/visits/drop": 0.3,
    "edges/visits/persistence": 2,
    "edges/near/cap": 1,
    "edges/near/cap_order": "random",
    "edges/hop:2/drop": 0.3,
    "edges/rewire/drop": 0.2,
}
TIED = {
    "derive/project:visits:sources:person:person": True,
    "derive/project_cap": 2,
    "derive/hops": 2,
    "derive/diffusion": "ppr",
    "derive/diffusion_top_k": 1,
    "edges/visits/cap": 1,
    "edges/visits/cap_order": "recent",
    "edges/hop:2/cap": 1,
    "edges/hop:2/cap_order": "recent",
    "relations/parallel": "merge",
}
WALKS = {"walks/bias": 0.5, "walks/fanout": 3, "walks/window": 6}
CAPS = {"coupling/scheme": "window", "coupling/window": 4, "coupling/cap": 1, "events/cap": 1}
JOINED = {"graphs/join": "single", "graphs/join_size": 2}
GENERATORS = (
    (np.random, ("default_rng", "Generator", "RandomState", "seed", "random", "rand", "choice", "permutation")),
    (torch, ("Generator", "manual_seed", "rand", "randn", "randint", "randperm", "multinomial", "bernoulli")),
)


class Unread:
    max_facts, max_working_bytes = LIMITS.max_facts, LIMITS.max_working_bytes

    @property
    def device(self):
        raise AssertionError("a derivation read the device")


class Held:
    def __init__(self, arrays):
        self.arrays = arrays

    def numpy(self, key):
        return np.asarray(self.arrays[key])

    def column(self, key, index):
        return np.ascontiguousarray(self.numpy(key)[:, index])


def stated(registry, values, graph):
    started = time.perf_counter()
    entries, rest = support.pipelined(registry, values, support.SEVEN)
    assert rest == {}
    result = pipeline.resolved(registry, entries), specs.configured(
        specs.settings(registry, GROUPS), dict(WALKS, **CAPS, **graph)
    )
    LOG.debug("stated exit settings=%d seconds=%.4f", len(graph), time.perf_counter() - started)
    return result


def permuted(graph, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    arrays = dict(graph.arrays)
    place = rng.permutation(len(graph.kinds))
    for name in ("kinds", "entity_token", "entity_connected"):
        arrays[name] = np.empty_like(np.asarray(arrays[name]))
        arrays[name][place] = np.asarray(graph.arrays[name])
    for name in ENTITIES:
        held = np.asarray(arrays[name])
        arrays[name] = np.where(held >= 0, place[np.maximum(held, 0)], -1).astype(held.dtype)
    keys = [None] * len(place)
    for entity, key in enumerate(graph.keys):
        keys[place[entity]] = key
    for table, reference in (("x_table", "obs_ref"), ("value_table", "fact_ref")):
        rows = rng.permutation(len(arrays[table]))
        arrays[table] = np.empty_like(np.asarray(arrays[table]))
        arrays[table][rows] = np.asarray(graph.arrays[table])
        held = np.asarray(arrays[reference])
        arrays[reference] = np.where(held >= 0, rows[np.maximum(held, 0)], -1).astype(held.dtype)
    order = np.lexsort((arrays["obs_entity"], arrays["obs_step"]))
    for name in ("obs_entity", "obs_step", "obs_ref", "obs_anchor"):
        arrays[name] = np.asarray(arrays[name])[order]
    step, present = np.asarray(arrays["fact_step"]), np.asarray(arrays["fact_present"])
    edge = labelled(step, present, arrays["fact_code"], arrays["fact_source"], arrays["fact_target"])
    order = np.lexsort((np.arange(len(edge)), rng.random(len(edge))[edge], ~present, step))
    for name in FACTS:
        arrays[name] = np.asarray(arrays[name])[order]
    result = Dense(graph.name, keys, graph.steps, **arrays)
    LOG.debug("permuted exit graph=%s seconds=%.4f", graph.name, time.perf_counter() - started)
    return result


def variant(members, registry, steps, spec, cuts, limits):
    started = time.perf_counter()
    derivation = Derivation.begun(members, registry, steps, spec, GROUPS, transform.SEED, limits, None, 0)
    for high in cuts + [sum(member.steps for member in members)]:
        derivation = derivation.continued(high, None)
    result = derivation.derived(), derivation.arrays()
    LOG.info("variant exit graphs=%d cuts=%d seconds=%.3f", len(members), len(cuts), time.perf_counter() - started)
    return result


def keyed(found, registry):
    started = time.perf_counter()
    dense, arrays = found
    keys = Keys(dense, registry)
    held = variant_part(arrays, keys, list(arrays))
    result = dict(dense_keyed(dense, keys), **{"variant/" + name: value for name, value in held.items()})
    LOG.debug("keyed exit arrays=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def derived(members, registry, steps, spec, cuts, limits):
    started = time.perf_counter()
    result = keyed(variant(members, registry, steps, spec, cuts, limits), registry)
    LOG.debug("derived exit graphs=%d seconds=%.4f", len(members), time.perf_counter() - started)
    return result


def levels(tree):
    started = time.perf_counter()
    found = []
    while tree is not None:
        found.append(("x", tree["x"].segments[0][2][0][1]))
        found += [(name, tree[name]) for name in ("mask", "gap", "relation") if name in tree]
        tree = tree.get("child")
    LOG.debug("levels exit arrays=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def sampled(found, registry, order):
    started = time.perf_counter()
    dense, arrays = found
    keys = Keys(dense, registry)
    spaces, held = Spaces(arrays, keys), Held(arrays)
    spec = {"neighbors/order": order, "neighbors/decay": 0.5}
    window = SimpleNamespace(graph=0, held=held, start=0, registry=registry, cfg=dict(graph=spec))
    owner, kinds = arrays["member_row"], arrays["row_kind"]
    pair = inputs.paired(registry, kinds, owner, arrays["member_name"])
    b = dict(
        member_row=owner,
        member_entity=arrays["member_entity"],
        row_steps=arrays["row_step"].astype(np.int64),
        row_kind=kinds,
        pair=pair,
        partner=inputs.partnered(pair, len(owner)),
    )
    history = inputs.history_inputs(window, b, SAMPLED, True, None)
    cutoffs = b["row_steps"][owner] * 2 + 1
    tree = inputs.tree(
        0, held, b["member_entity"].astype(np.int64), cutoffs, 0, SAMPLED, 2, registry, spec, transform.SEED
    )
    x, task = (piece[1] for piece in history["history_x"].segments[0][2])
    paired = inputs.history_inputs(window, b, PAIRED, False, None)
    result = dict(
        history=rows(
            spaces.member,
            through(keys.entities, history["history_ids"]),
            through(spaces.observations, x),
            through(keys.tasks, task),
            *(history[name] for name in ("history_mask", "history_gap", "history_relation", "occurrence", "structure")),
        ),
        common=rows(
            spaces.rows,
            through(keys.entities, paired["common_ids"]),
            *(paired[name] for name in ("common_frequency", "common_stats", "pair_gap", "pair_mask")),
        ),
        tree=rows(
            spaces.member,
            *(
                (through(spaces.observations, value) if name == "x" else value).reshape(len(owner), -1)
                for name, value in levels(tree)
            ),
        ),
    )
    full, shared = int(history["history_mask"].all(1).sum()), int(((paired["common_ids"] >= 0).sum(1) > 1).sum())
    LOG.info(
        "sampled exit order=%s full=%d shared=%d seconds=%.3f", order, full, shared, time.perf_counter() - started
    )
    return result, full, shared


@pytest.fixture(scope="module")
def drawn(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(opened)
    views = [support.shown(opened, index).dense() for index in range(4)]
    cfg = support.configuration(
        opened.registry, dict(view="snapshot"), dict(support.several(opened.registry), **JOINED), 5, SPLITS
    )
    steps, spec = stated(registry, DRAWN, {})
    reference = derived(views[:1], registry, steps, spec, [], LIMITS)
    return registry, views, reference, (opened.viewed(cfg).graphs, opened.groups(cfg))


@pytest.fixture(scope="module")
def unlinked(drawn):
    registry, views, _, _ = drawn
    moved = [permuted(graph, seed) for seed, graph in enumerate(views)]
    separate = dict(JOINED, **{"graphs/connect/" + kind: False for kind in registry.nodes})
    steps, spec = stated(registry, DRAWN, separate)
    whole = variant(views[:2], registry, steps, spec, [], LIMITS)
    turned = variant(moved[1::-1], registry, steps, spec, [5, views[1].steps + 2], LIMITS)
    return moved, whole, turned


def test_permuting_entities_and_graphs_leaves_every_derived_array_equal_after_sorting_by_key(drawn, unlinked):
    started = time.perf_counter()
    registry, views, reference, (entries, groups) = drawn
    moved, whole, turned = unlinked
    graphs = views[:2]
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(moved[:1], registry, steps, spec, [4, 9], LIMITS), reference) == []
    assert len(reference["variant/walks"]) and registry.code("rewire") in reference["facts"][:, 2]
    assert differing(keyed(turned, registry), keyed(whole, registry)) == []
    steps, spec = stated(registry, DRAWN, dict(JOINED, **{"graphs/connect_fraction": 0.5}))
    linked = derived(graphs, registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[:2], registry, steps, spec, [graphs[0].steps + 3], LIMITS), linked) == []
    assert set(linked["entities"][:, -1].tolist()) == {0, 1}
    order = [2, 0, 3, 1]
    keys = [(0, 1)] * len(entries)
    turned = grouped([dict(entries[old], index=new) for new, old in enumerate(order)], keys, 2)
    assert grouped(entries, keys, 2) == groups
    assert [[entries[order[new]]["name"] for new in group] for group in turned] == [
        [entries[index]["name"] for index in group] for group in groups
    ]
    last = derived([views[index] for index in groups[-1]], registry, steps, spec, [], LIMITS)
    placed = [moved[order[new]] for new in turned[-1]]
    assert differing(derived(placed, registry, steps, spec, [5], LIMITS), last) == []
    steps, spec = stated(registry, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    fosr = derived(graphs[1:], registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[1:2], registry, steps, spec, [7], LIMITS), fosr) == []
    steps, spec = stated(registry, TIED, {})
    tied = derived(graphs[:1], registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[:1], registry, steps, spec, [4], LIMITS), tied) == []
    LOG.info("permuted draws exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.parametrize("order", ["recent", "uniform"])
def test_the_neighbours_a_batch_reads_are_equal_when_entities_are_renumbered_or_graphs_reordered(
    drawn, unlinked, order
):
    registry = drawn[0]
    _, whole, turned = unlinked
    found, full, shared = sampled(whole, registry, order)
    assert full and shared and differing(sampled(turned, registry, order)[0], found) == []


def test_no_draw_reads_a_random_generator_or_a_device(drawn, monkeypatch):
    registry, views, reference, _ = drawn
    for module, names in GENERATORS:
        for name in names:
            monkeypatch.setattr(module, name, lambda *args, name=name, **kwargs: pytest.fail("a draw used " + name))
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(views[:1], registry, steps, spec, [6], Unread()), reference) == []


@pytest.mark.gpu
def test_the_arrays_of_every_draw_are_equal_on_cpu_and_on_gpu(drawn):
    registry, views, reference, _ = drawn
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(views[:1], registry, steps, spec, [], GraphLimits("cuda")), reference) == []
    steps, spec = stated(registry, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    on = [derived(views[1:2], registry, steps, spec, [], GraphLimits(device)) for device in ("cpu", "cuda")]
    assert differing(on[1], on[0]) == []
